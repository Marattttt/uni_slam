#include "map_stage.hpp"

#include <spdlog/spdlog.h>

#include <expected>
#include <flat_map>
#include <memory>
#include <numbers>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "associate_landmarks.hpp"
#include "build_factors.hpp"
#include "compute/stage.hpp"
#include "feed_loop_closure.hpp"
#include "filter_keyframe.hpp"
#include "loop_closure/global_ba.hpp"
#include "loop_closure/merge.hpp"
#include "map/utils.hpp"
#include "map_common.hpp"
#include "models.hpp"
#include "pose_estimate_v2/util/camera.hpp"
#include "submit_updates.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;
namespace utils = wslam::map::utils;

#define LOG_ID "[Mapping v2 stage]"

namespace {
// The whole run as a global bundle adjustment problem, started from the
// loop-corrected map: every keyframe the smoother left a velocity and bias
// for, every recorded IMU edge, every landmark with a corrected position,
// plus the loops' PnP inliers as observations of the candidates' landmarks
// from the query keyframes
std::expected<loop_closure::GlobalBaInput, std::string> MakeGlobalBaInput(
    const AnyBag& storage, const SharedBindings& shared,
    const loop_closure::LoopClosureWorker& worker, const gtsam::Values& window,
    const MapSnapshot& corrected, size_t& loop_observations) {
    auto camera = v2::util::ReadCameraParams(storage, 0);
    if (!camera) {
        return std::unexpected(std::move(camera).error());
    }
    const auto imu = storage.getPtr<data::IMUSensorParams>(
        ResourceIdentifier::ImuParamsName);
    if (!imu || shared.cam_calibration == nullptr) {
        return std::unexpected("no IMU parameters or camera calibration");
    }

    loop_closure::GlobalBaInput input;
    input.calibration = shared.cam_calibration;
    input.imu_params = utils::CreateImuCalibration(
        *camera, *imu.value(), shared.gravity_world,
        BuildFactorsPass::Opts{}.integration_sigma);

    for (const KeyframePose& keyframe : corrected.keyframes) {
        const v2::mapping::PoseId id{keyframe.id.v};
        std::optional<NavigationEstimate> navigation;
        if (window.exists(Velocity(id)) && window.exists(Bias(id))) {
            navigation = NavigationEstimate{
                .velocity = window.at<gtsam::Vector3>(Velocity(id)),
                .bias = window.at<gtsam::imuBias::ConstantBias>(Bias(id)),
            };
        } else if (const auto archived = shared.archived_navigation.find(id);
                   archived != shared.archived_navigation.end()) {
            navigation = archived->second;
        }
        if (!navigation) {
            continue;
        }
        // The velocity turns with the keyframe's correction
        const gtsam::Rot3 turn
            = worker.graph().smoothCorrectionFor(id.v).rotation();
        input.keyframes.push_back(loop_closure::GlobalBaInput::Keyframe{
            .id = id.v,
            .pose = gtsam::Pose3(gtsam::Rot3(keyframe.R_world_cam),
                                 gtsam::Point3(keyframe.t_world_cam)),
            .velocity = turn.rotate(gtsam::Point3(navigation->velocity)),
            .bias = navigation->bias,
        });
    }

    for (const auto& [to, edge] : shared.imu_edges) {
        input.imu_edges.push_back(loop_closure::GlobalBaInput::ImuEdge{
            .from = edge.from.v,
            .to = to.v,
            .from_ts = edge.from_ts,
            .to_ts = edge.to_ts,
            .samples = edge.samples,
        });
    }

    std::flat_map<uint64_t, std::vector<std::pair<uint64_t, Eigen::Vector2d>>>
        loop_views;
    const gtsam::Cal3_S2& k = *shared.cam_calibration;
    for (const auto& loop : worker.loops()) {
        for (const auto& observation : loop.observations) {
            loop_views[observation.landmark].emplace_back(
                loop.edge.query,
                Eigen::Vector2d{k.fx() * observation.query_ray.x() + k.px(),
                                k.fy() * observation.query_ray.y() + k.py()});
        }
    }

    for (const LandmarkEstimate& landmark : corrected.landmarks) {
        const auto recorded = shared.landmark_observations.find(
            v2::mapping::LandmarkId{landmark.id.v});
        if (recorded == shared.landmark_observations.end()) {
            continue;
        }
        loop_closure::GlobalBaInput::Landmark entry{
            .id = landmark.id.v, .position = landmark.position_world};
        for (const auto& [pose, pixel] : recorded->second) {
            entry.observations.emplace_back(pose.v, pixel);
        }
        if (const auto views = loop_views.find(landmark.id.v);
            views != loop_views.end()) {
            entry.observations.insert(entry.observations.end(),
                                      views->second.begin(),
                                      views->second.end());
            loop_observations += views->second.size();
        }
        input.landmarks.push_back(std::move(entry));
    }

    return input;
}

// Runs the global bundle adjustment on the loop-corrected map. Returns the
// adjusted map, or nullopt when it was rejected; report says what happened
std::optional<MapSnapshot> AdjustGlobally(
    const AnyBag& storage, const SharedBindings& shared,
    const loop_closure::LoopClosureWorker& worker, const gtsam::Values& window,
    const MapSnapshot& corrected, GlobalBaReport& report) {
    loop_closure::GlobalBaOptions options;
    options.max_iterations
        = utils::EnvU32("WSLAM_GBA_MAX_ITERS", options.max_iterations);
    options.min_observations = utils::EnvU32(
        "WSLAM_GBA_MIN_OBS", static_cast<uint32_t>(options.min_observations));

    auto input = MakeGlobalBaInput(storage, shared, worker, window, corrected,
                                   report.loop_observations);
    if (!input) {
        report.outcome = "no input: " + input.error();
        return std::nullopt;
    }
    report.keyframes = input->keyframes.size();

    auto result = loop_closure::RunGlobalBa(input.value(), options);
    if (!result) {
        report.outcome = "rejected: " + result.error();
        return std::nullopt;
    }
    report.accepted = true;
    report.outcome = "accepted";
    report.imu_factors = result->imu_factors;
    report.landmarks = result->landmarks.size();
    report.observations = result->projection_factors;
    report.outliers_removed = result->outliers_removed;
    report.iterations = result->iterations;
    report.initial_error = result->initial_error;
    report.final_error = result->final_error;
    report.path_length_ratio = result->path_length_ratio;
    report.seconds = result->seconds;

    // Keyframes and landmarks the adjustment left out keep their corrected
    // estimate
    MapSnapshot adjusted = corrected;
    const std::flat_map<uint64_t, gtsam::Pose3> poses(std::from_range,
                                                      std::move(result->poses));
    for (KeyframePose& keyframe : adjusted.keyframes) {
        if (const auto it = poses.find(keyframe.id.v); it != poses.end()) {
            keyframe.R_world_cam = it->second.rotation().matrix();
            keyframe.t_world_cam = it->second.translation();
        }
    }
    const std::flat_map<uint64_t, Eigen::Vector3d> points(
        std::from_range, std::move(result->landmarks));
    for (LandmarkEstimate& landmark : adjusted.landmarks) {
        if (const auto it = points.find(landmark.id.v); it != points.end()) {
            landmark.position_world = it->second;
        }
    }
    return adjusted;
}

// Moves the smoother's final snapshot to MapSnapshotVioName and publishes
// its loop-corrected copy under MapSnapshotName, with the loop closure report.
// With a global bundle adjustment that is accepted, the corrected copy moves
// to MapSnapshotPgoName and MapSnapshotName holds the adjusted map
std::optional<std::string> MergeLoopClosure(
    AnyBag& storage, const SharedBindings& shared,
    const loop_closure::LoopClosureWorker& worker, const gtsam::Values& window,
    bool global_ba) {
    auto vio = storage.get<MapSnapshot>(ResourceIdentifier::MapSnapshotName);
    if (!vio) {
        return "no final snapshot to merge loop closure into";
    }

    const auto anchor_of
        = [&shared](uint64_t landmark) -> std::optional<uint64_t> {
        const auto it
            = shared.landmark_anchor.find(wslam::map::LandmarkId{landmark});
        if (it == shared.landmark_anchor.end()) {
            return std::nullopt;
        }
        return it->second.v;
    };

    loop_closure::MergeStats stats;
    MapSnapshot corrected = loop_closure::ApplyCorrections(
        vio.value(), worker.graph(), anchor_of, &stats);
    LoopClosureReport report = loop_closure::MakeReport(worker);

    spdlog::info(LOG_ID
                 " loop closure ({}): {} keyframes, {} nodes, {} queries, {} "
                 "loops; corrected {} keyframes and {} landmarks ({} without "
                 "an anchor)",
                 report.model, report.keyframes, report.nodes, report.queries,
                 report.loops.size(), stats.keyframes_corrected,
                 stats.landmarks_corrected, stats.landmarks_unanchored);

    storage.set(ResourceIdentifier::MapSnapshotVioName, std::move(vio).value());

    if (global_ba) {
        GlobalBaReport ba_report;
        std::optional<MapSnapshot> adjusted = AdjustGlobally(
            storage, shared, worker, window, corrected, ba_report);
        spdlog::info(LOG_ID " global BA {}", ba_report.outcome);
        report.global_ba = std::move(ba_report);
        if (adjusted) {
            storage.set(ResourceIdentifier::MapSnapshotPgoName,
                        std::move(corrected));
            corrected = std::move(adjusted).value();
        }
    }

    storage.set(ResourceIdentifier::MapSnapshotName, std::move(corrected));
    storage.set(ResourceIdentifier::LoopClosureReportName, std::move(report));

    return std::nullopt;
}
}  // namespace

MappingStage v2::mapping::CreateMappingStage(
    compute::Compute& compute, AnyBag& storage, const WslamConfig& config,
    std::shared_ptr<loop_closure::LoopClosureWorker> loop_closure) {
    spdlog::info(LOG_ID " Constructing mapping stage");

    // Owned here; the passes hold it by reference. flush_async keeps it alive
    // as long as Compute holds the passes
    auto shared = std::make_shared<SharedBindings>();

    FilterKeyframeConfig filter_config{};
    filter_config.gravity_magnitude = utils::EnvDouble(
        "WSLAM_GRAVITY_MAGNITUDE", filter_config.gravity_magnitude);
    filter_config.zupt_max_score = utils::EnvDouble(
        "WSLAM_INIT_ZUPT_SCORE", filter_config.zupt_max_score);
    filter_config.initial_velocity_sigma = utils::EnvDouble(
        "WSLAM_INIT_VELOCITY_SIGMA", filter_config.initial_velocity_sigma);
    filter_config.initial_velocity_sigma_per_s
        = utils::EnvDouble("WSLAM_INIT_VELOCITY_SIGMA_PER_S",
                           filter_config.initial_velocity_sigma_per_s);

    BuildFactorsPass::Opts build_opts{
        .gate_min_rotation_rad = filter_config.min_rotation_rad,
        .gate_min_parallax_px = filter_config.min_parallax_px,
    };
    build_opts.prior_pose_tilt_sigma_rad
        = utils::EnvDouble(
              "WSLAM_INIT_TILT_SIGMA_DEG",
              build_opts.prior_pose_tilt_sigma_rad * 180.0 / std::numbers::pi)
          * std::numbers::pi / 180.0;
    build_opts.prior_accel_bias_sigma = utils::EnvDouble(
        "WSLAM_INIT_ACCEL_BIAS_SIGMA", build_opts.prior_accel_bias_sigma);

    spdlog::info(
        LOG_ID
        " IMU init: |g| {:.4f}, zero-velocity score < {:.2f}, "
        "velocity sigma {:.3f} + {:.3f}/s m/s, tilt sigma {:.2f} deg, "
        "accel bias sigma {:.3f}",
        filter_config.gravity_magnitude, filter_config.zupt_max_score,
        filter_config.initial_velocity_sigma,
        filter_config.initial_velocity_sigma_per_s,
        build_opts.prior_pose_tilt_sigma_rad * 180.0 / std::numbers::pi,
        build_opts.prior_accel_bias_sigma);

    SubmitSmootherPass::Opts submit_opts{};
    // Headless runs only read the final snapshot (flush() publishes it); the
    // GUI would read one every keyframe
    submit_opts.snapshot_every_n_keyframes
        = config.enable_gui ? 1U : utils::EnvU32("WSLAM_SNAPSHOT_EVERY_N", 0U);
    submit_opts.smoother.window_keyframes = utils::EnvU32(
        "WSLAM_SMOOTHER_WINDOW", submit_opts.smoother.window_keyframes);
    submit_opts.smoother.max_iterations = utils::EnvU32(
        "WSLAM_SMOOTHER_MAX_ITERS", submit_opts.smoother.max_iterations);

    spdlog::info(LOG_ID
                 " back-end config: window={} keyframes, max_iterations={}, "
                 "snapshot_every_n={}",
                 submit_opts.smoother.window_keyframes,
                 submit_opts.smoother.max_iterations,
                 submit_opts.snapshot_every_n_keyframes);

    auto stage
        = std::make_unique<compute::Stage>("Mapping v2", &compute.getPerf());

    auto submit
        = std::make_unique<SubmitSmootherPass>(storage, *shared, submit_opts);
    // The drain pass and the flushable reach the submit pass through this
    // pointer; all three live inside Compute together
    SubmitSmootherPass* submit_ptr = submit.get();

    stage->add_pass(
        std::make_unique<FilterKeyframePass>(storage, *shared, filter_config));
    stage->add_pass(std::make_unique<DrainSmootherPass>(*submit_ptr));
    stage->add_pass(std::make_unique<AssociateLandmarksPass>(
        storage, *shared,
        AssociateLandmarksPass::Opts{
            .association_tolerance_px = filter_config.association_tolerance_px,
        }));
    stage->add_pass(
        std::make_unique<BuildFactorsPass>(storage, *shared, build_opts));
    stage->add_pass(std::move(submit));

    if (!loop_closure) {
        return MappingStage{
            .stage = std::move(stage),
            .flush_async
            = [shared, submit_ptr]() -> std::optional<std::string> {
                return submit_ptr->flush();
            },
        };
    }

    // Export-time global bundle adjustment after the pose graph merge
    const bool global_ba = utils::EnvU32("WSLAM_GLOBAL_BA", 1U) != 0;
    spdlog::info(LOG_ID " global bundle adjustment at export: {}",
                 global_ba ? "on" : "off");

    auto feed
        = std::make_unique<FeedLoopClosurePass>(storage, *shared, loop_closure);
    FeedLoopClosurePass* feed_ptr = feed.get();
    stage->add_pass(std::move(feed));

    return MappingStage{
        .stage = std::move(stage),
        .flush_async = [shared, submit_ptr, feed_ptr, loop_closure, &storage,
                        global_ba]() -> std::optional<std::string> {
            if (auto err = submit_ptr->flush()) {
                return err;
            }

            feed_ptr->flush();
            if (auto err = loop_closure->finish()) {
                return "loop closure: " + std::move(err).value();
            }

            return MergeLoopClosure(storage, *shared, *loop_closure,
                                    submit_ptr->latestValues(), global_ba);
        },
    };
}
