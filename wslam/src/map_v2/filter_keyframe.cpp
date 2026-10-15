#include "filter_keyframe.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <flat_map>
#include <format>
#include <string>
#include <utility>

#include "assert.hpp"
#include "common.hpp"
#include "compute/stage.hpp"
#include "map/map_helpers_lods.hpp"
#include "map/utils.hpp"
#include "pose_estimate_v2/util/camera.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;
namespace utils = wslam::map::utils;

#define LOG_ID "[Filter Keyframe v2 pass]"

namespace {
// Median depth of the triangulated landmarks in the current camera, in the
// units of the unit translation; 0 without landmarks
double MedianDepth(const v2::KeypointTriangulationResult& triangulation) {
    std::vector<double> depths;
    depths.reserve(triangulation.landmarks.size());
    for (const v2::KeypointLandmark& landmark : triangulation.landmarks) {
        depths.push_back(landmark.position_cam_curr.z());
    }
    if (depths.empty()) {
        return 0.0;
    }
    const auto middle
        = depths.begin() + static_cast<std::ptrdiff_t>(depths.size() / 2);
    std::nth_element(depths.begin(), middle, depths.end());
    return *middle;
}
}  // namespace

std::string FilterKeyframePass::getId() const { return LOG_ID; }

std::optional<std::string> FilterKeyframePass::initialize() {
    spdlog::info(LOG_ID " Initializing");

    auto camera = v2::util::ReadCameraParams(storage_, config_.camera_index);
    if (!camera) {
        return std::move(camera).error();
    }
    camera_ = std::move(camera).value();

    return std::nullopt;
}

std::optional<std::string> FilterKeyframePass::execute() {
    spdlog::info(LOG_ID " Executing");
    WSLAM_ASSERT(camera_.has_value(), "executing before initialize()");

    // Until the first keyframe the reference advances every frame, so the
    // bootstrap baseline stays frame-to-frame short and the first keyframe
    // lands at motion onset; afterwards only an accepted keyframe advances it,
    // so the geometry spans exactly the previous-keyframe -> current edge
    const bool had_origin = shared_.keyframes_processed > 0;
    if (!had_origin) {
        storage_.set(ResourceIdentifier::FeatureReferenceAdvanceName, true);
    }

    // Taken whatever happens, so neither outlives its frame
    const bool skipped
        = storage_.take<bool>(ResourceIdentifier::FrameSkippedName)
              .value_or(false);
    const auto prediction = storage_.take<KeyframeMotionPrediction>(
        ResourceIdentifier::KeyframeMotionPredictionName);

    auto triangulation = storage_.take<KeypointTriangulationResult>(
        ResourceIdentifier::KeypointTriangulationName);
    auto ransac = storage_.take<KeypointRansacResult>(
        ResourceIdentifier::KeypointRansacName);
    if (skipped) {
        spdlog::debug(LOG_ID " The prefilter skipped this frame");
        return compute::kStageStopExecution;
    }
    if (!triangulation || !ransac) {
        spdlog::warn(LOG_ID " No pose estimate for this frame");
        return compute::kStageStopExecution;
    }

    if (!triangulation->stats.pose_recovered) {
        spdlog::warn(LOG_ID " No pose recovered");
        return compute::kStageStopExecution;
    }
    if (triangulation->landmarks.size() < config_.min_landmarks) {
        spdlog::warn(LOG_ID " Not enough landmarks: {} < {}",
                     triangulation->landmarks.size(), config_.min_landmarks);
        return compute::kStageStopExecution;
    }
    if (ransac->inliers.empty()) {
        spdlog::warn(LOG_ID " RANSAC reports no inliers");
        return compute::kStageStopExecution;
    }

    MatchInfo info = collectMatches(*ransac);

    if (prediction) {
        // Grep-friendly: how the frame prefilter's IMU prediction compares
        // with what the gate measures
        spdlog::info(
            LOG_ID
            " [prefilter] dt {:.3f} s: predicted rotation {:.4f} (<= {:.4f}) "
            "rad, parallax {:.2f} (<= {:.2f}) px; measured {:.4f} rad, "
            "{:.2f} px",
            prediction->dt_s, prediction->rotation_rad,
            prediction->rotation_upper_rad, prediction->parallax_px,
            prediction->parallax_upper_px,
            wslam::map::util::ComputeRotationAngle(triangulation->rotation),
            info.median_parallax);
    }

    if (!shouldAccept(*triangulation, info)) {
        return compute::kStageStopExecution;
    }

    KeyframeCandidate candidate = initializeCandidate(*triangulation);
    candidate.median_depth_unit = MedianDepth(*triangulation);

    const auto curr_kf_ts
        = storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName);
    if (!curr_kf_ts) {
        return "could not get current keyframe timestamp";
    }
    candidate.curr_kf_ts = curr_kf_ts.value();
    candidate.prev_kf_ts = shared_.last_kf_ts;

    if (auto err = processImu(candidate)) {
        return "imu: " + std::move(err).value();
    }

    candidate.matches = std::move(info.matches);

    spdlog::info(LOG_ID
                 " Accepted x{}: {} matches ({} tracked, {} pixel "
                 "mismatches), median parallax {:.2f} px",
                 candidate.pose_id.v, candidate.matches.size(), info.tracked,
                 info.pixel_mismatches, info.median_parallax);

    shared_.keyframes_processed++;
    shared_.last_pose = candidate.pose_id;
    shared_.last_kf_ts = candidate.curr_kf_ts;
    shared_.pose_timestamps.insert_or_assign(candidate.pose_id,
                                             candidate.curr_kf_ts);
    shared_.predicted_values.insert(
        Pose(candidate.pose_id),
        gtsam::Pose3{gtsam::Rot3{candidate.R_world_cam},
                     gtsam::Point3{candidate.t_world_cam}});

    // The prior describes the previous keyframe; the build pass publishes
    // this one's
    storage_.erase(ResourceIdentifier::KeyframeMotionPriorName);

    // Accepted keyframe: it becomes the model's reference next frame
    storage_.set(ResourceIdentifier::FeatureReferenceAdvanceName, true);
    storage_.set(ResourceIdentifier::KeyframeCandidateName,
                 std::move(candidate));

    return std::nullopt;
}

auto FilterKeyframePass::collectMatches(
    const KeypointRansacResult& ransac) const -> MatchInfo {
    const data::CamSensorParams& camera = *camera_;

    MatchInfo info;
    std::vector<double> parallaxes;
    parallaxes.reserve(ransac.inliers.size());

    for (const KeypointCorrespondence& inlier : ransac.inliers) {
        const double parallax = (inlier.curr_px - inlier.prev_px).norm();

        // The accept/bootstrap decision reads the median over ALL inliers, so
        // a near-stationary frame (inliers clustered at zero parallax) reads
        // ~0 and is rejected; a median over only the high-parallax pairs
        // let a few spurious matches anchor degenerate short-baseline
        // keyframes
        parallaxes.push_back(parallax);

        // Landmarks still need a real baseline for a usable depth
        if (parallax < config_.landmark_parallax_filter_px) {
            continue;
        }

        CandidateMatch match{
            .prev_index = inlier.prev_index,
            .curr_index = inlier.curr_index,
            .prev_pixel_observed = inlier.prev_px,
            .curr_pixel_observed = inlier.curr_px,
            .prev_pixel = v2::util::NormalisedToPixel(inlier.prev_ray, camera),
            .curr_pixel = v2::util::NormalisedToPixel(inlier.curr_ray, camera),
        };

        switch (LookupTrack(shared_, inlier.prev_index, inlier.prev_px,
                            config_.association_tolerance_px)
                    .outcome) {
            case TrackLookup::Tracked:
                ++info.tracked;
                break;
            case TrackLookup::PixelMismatch:
                ++info.pixel_mismatches;
                break;
            case TrackLookup::Untracked:
                break;
        }

        info.matches.push_back(std::move(match));
    }

    const auto mid = parallaxes.begin()
                     + static_cast<std::ptrdiff_t>(parallaxes.size() / 2);
    std::nth_element(parallaxes.begin(), mid, parallaxes.end());
    info.median_parallax = *mid;

    return info;
}

bool FilterKeyframePass::shouldAccept(
    const KeypointTriangulationResult& triangulation,
    const MatchInfo& info) const {
    const double rotation
        = wslam::map::util::ComputeRotationAngle(triangulation.rotation);

    if (shared_.keyframes_processed == 0) {
        // The first keyframe seeds gravity and gyroscope bias from a full
        // stationary IMU window, so it waits for a window's worth of samples
        const auto imu_buffer = storage_.getPtr<std::vector<data::IMUReading>>(
            ResourceIdentifier::GetImuVecName());
        const size_t imu_samples
            = imu_buffer.has_value() ? imu_buffer.value()->size() : 0;
        if (imu_samples < config_.gravity_window_samples) {
            spdlog::warn(LOG_ID
                         " Bootstrap: {} IMU samples < {} needed for the "
                         "gravity window; waiting",
                         imu_samples, config_.gravity_window_samples);
            return false;
        }

        if (info.median_parallax < config_.bootstrap_min_parallax_px) {
            spdlog::warn(
                LOG_ID
                " Bootstrap: parallax median {:.2f} < {:.2f}; not anchoring",
                info.median_parallax, config_.bootstrap_min_parallax_px);
            return false;
        }

        // The first keyframe is the origin: it has no edge to gate, so it
        // lands at motion onset, where its zero-velocity prior and the
        // gravity of the stationary window hold. The small-motion test below
        // reads frame-to-frame motion here (the reference advances every
        // frame before origin) and held the bootstrap back until fast motion,
        // ~28 s into V1_01_easy's flight
        spdlog::info(LOG_ID
                     " Bootstrapping on parallax median {:.2f} and "
                     "rotation {:.2f}",
                     info.median_parallax, rotation);
        return true;
    }

    const bool is_small_motion
        = rotation < config_.min_rotation_rad
          && info.median_parallax < config_.min_parallax_px;
    if (is_small_motion) {
        spdlog::info(LOG_ID
                     " Motion too small (rotation {:.3f} rad, parallax median "
                     "{:.2f} px)",
                     rotation, info.median_parallax);
        return false;
    }

    // Counted against the landmarks of the last keyframe before the pending
    // smoother update is drained; retirements there only turn tracked
    // matches into new ones, so this undercounts
    const size_t new_matches = info.matches.size() - info.tracked;
    if (new_matches < config_.min_new_landmarks) {
        spdlog::info(LOG_ID " Not enough new landmarks: {} < {}", new_matches,
                     config_.min_new_landmarks);
        return false;
    }

    return true;
}

KeyframeCandidate FilterKeyframePass::initializeCandidate(
    const KeypointTriangulationResult& triangulation) const {
    KeyframeCandidate candidate;
    candidate.pose_id = PoseId{shared_.keyframes_processed + 1};
    candidate.R_rel = triangulation.rotation;
    candidate.t_rel = triangulation.translation;

    if (shared_.keyframes_processed == 0) {
        return candidate;
    }

    WSLAM_ASSERT(shared_.last_pose.v > 0, "a keyframe was accepted before");
    candidate.prev_pose_id = shared_.last_pose;

    // p_curr = R * p_prev + t between the camera frames, so with
    // T_w_prev = (R_wp, t_wp):
    //   R_wc = R_wp * R^T
    //   t_wc = t_wp - R_wc * t
    const auto& prev_pose = shared_.predicted_values.at<gtsam::Pose3>(
        Pose(candidate.prev_pose_id));
    const Eigen::Matrix3d r_wp = prev_pose.rotation().matrix();
    const Eigen::Vector3d t_wp = prev_pose.translation();
    candidate.R_world_cam = r_wp * triangulation.rotation.transpose();
    candidate.t_world_cam
        = t_wp - candidate.R_world_cam * triangulation.translation;

    return candidate;
}

std::optional<std::string> FilterKeyframePass::processImu(
    KeyframeCandidate& candidate) const {
    auto imu_buffer = storage_.take<std::vector<data::IMUReading>>(
        ResourceIdentifier::GetImuVecName());
    if (!imu_buffer) {
        return "could not take the IMU buffer";
    }

    if (candidate.isFirstKeyframe()) {
        if (auto err = initializeFirstFrame(imu_buffer.value(),
                                            candidate.curr_kf_ts)) {
            return "initializing first keyframe: " + std::move(err).value();
        }
    }

    const uint64_t from = candidate.prev_kf_ts;
    const uint64_t to = candidate.curr_kf_ts;

    auto first = std::ranges::find_if(imu_buffer.value(), [&](const auto& imu) {
        return imu.timestamp >= from;
    });
    auto last = std::find_if(first, imu_buffer->end(), [&](const auto& imu) {
        return imu.timestamp >= to;
    });

    // Samples after this keyframe stay for the next one
    std::vector<data::IMUReading> remainder{last, imu_buffer->end()};
    candidate.imu = std::vector<data::IMUReading>{first, last};
    storage_.set(ResourceIdentifier::GetImuVecName(), std::move(remainder));

    return std::nullopt;
}

std::optional<std::string> FilterKeyframePass::initializeFirstFrame(
    std::span<const data::IMUReading> imu_readings,
    uint64_t keyframe_ts) const {
    const auto imu_params = storage_.getPtr<data::IMUSensorParams>(
        ResourceIdentifier::ImuParamsName);
    if (!imu_params) {
        return "no IMU parameters";
    }

    const auto window = utils::FindStationaryWindow(
        imu_readings, config_.gravity_window_samples,
        config_.max_score_for_stationary);
    if (!window) {
        return "finding stationary window: " + window.error();
    }

    const utils::StartupImuState startup = utils::RecoverStartupImuState(
        window.value(), *camera_, *imu_params.value(),
        config_.gravity_magnitude);
    const uint64_t quiet_end = window->samples.back().timestamp;

    // The body is still at rest at the end of the last quiet window, which
    // may come after the quietest one (motors spinning up): the first
    // keyframe's velocity is integrated from there rather than assumed zero
    const auto before_keyframe = std::ranges::find_if(
        imu_readings, [keyframe_ts](const data::IMUReading& reading) {
            return reading.timestamp > keyframe_ts;
        });
    const auto last_rest = utils::FindLastStationaryWindow(
        {imu_readings.begin(), before_keyframe}, config_.gravity_window_samples,
        config_.zupt_max_score);
    const uint64_t rest_end
        = last_rest ? std::max(quiet_end, last_rest->samples.back().timestamp)
                    : quiet_end;

    // Both windows' gravity is in the camera frame at their end; the world
    // frame is this keyframe's camera, which may have turned since
    const utils::PropagatedRest at_rest_end = utils::PropagateFromRest(
        startup.gravity_cam, imu_readings, quiet_end, rest_end, startup.bias,
        *camera_, *imu_params.value());
    const utils::PropagatedRest at_keyframe = utils::PropagateFromRest(
        at_rest_end.gravity_cam, imu_readings, rest_end, keyframe_ts,
        startup.bias, *camera_, *imu_params.value());

    const Eigen::Vector3d& gravity = at_keyframe.gravity_cam;
    const double propagated_rad = std::acos(std::clamp(
        startup.gravity_cam.normalized().dot(gravity.normalized()), -1.0, 1.0));
    const double integrated_s
        = static_cast<double>(keyframe_ts - rest_end) * 1e-9;

    shared_.gravity_world = gravity;
    shared_.last_bias = startup.bias;
    shared_.last_velocity = at_keyframe.velocity_cam;
    shared_.initial_velocity_sigma
        = config_.initial_velocity_sigma
          + config_.initial_velocity_sigma_per_s * integrated_s;

    const auto to_seconds
        = [](std::integral auto ns) { return static_cast<double>(ns) * 1e-9; };
    const Eigen::Vector3d& accel_bias = startup.bias.accelerometer();
    const Eigen::Vector3d& gyro_bias = startup.bias.gyroscope();
    spdlog::info(
        LOG_ID
        " Initialized gravity from {} IMU samples; score:{:.3f} "
        "window:[{:.3f} - {:.3f}s] gravity=({:.3f}, {:.3f}, {:.3f}) "
        "(turned {:.2f} deg to the keyframe {:.3f} s later) "
        "gyro_bias=({:.4f}, {:.4f}, {:.4f}) accel_bias=({:.4f}, "
        "{:.4f}, {:.4f})",
        window->samples.size(), window->score,
        to_seconds(window->samples.front().timestamp), to_seconds(quiet_end),
        gravity.x(), gravity.y(), gravity.z(), propagated_rad * 180.0 / M_PI,
        to_seconds(keyframe_ts - quiet_end), gyro_bias.x(), gyro_bias.y(),
        gyro_bias.z(), accel_bias.x(), accel_bias.y(), accel_bias.z());
    spdlog::info(LOG_ID
                 " Initial velocity: at rest until {:.3f}s ({}), integrated "
                 "{:.3f} s to the keyframe: |v|={:.3f} m/s ({:.3f}, {:.3f}, "
                 "{:.3f}), prior sigma {:.3f} m/s",
                 to_seconds(rest_end),
                 last_rest
                     ? std::format("last window score {:.3f}", last_rest->score)
                     : std::string{"no later quiet window"},
                 integrated_s, at_keyframe.velocity_cam.norm(),
                 at_keyframe.velocity_cam.x(), at_keyframe.velocity_cam.y(),
                 at_keyframe.velocity_cam.z(), shared_.initial_velocity_sigma);

    return std::nullopt;
}
