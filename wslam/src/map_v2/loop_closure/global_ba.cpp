#include "global_ba.hpp"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <map>
#include <set>

#include "map/map_common.hpp"
#include "map/utils.hpp"

using wslam::v2::mapping::loop_closure::GlobalBaInput;
using wslam::v2::mapping::loop_closure::GlobalBaOptions;
using wslam::v2::mapping::loop_closure::GlobalBaResult;
using MapLandmarkId = wslam::map::LandmarkId;
using MapPoseId = wslam::map::PoseId;
namespace keys = wslam::map::GtsamIdentifiers;

#define LOG_ID "[Global BA]"

namespace {

using ProjectionFactor
    = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                     gtsam::Cal3_S2>;

struct Problem {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    size_t imu_factors = 0;
    size_t projection_factors = 0;
};

double PathLength(const gtsam::Values& values,
                  const std::vector<GlobalBaInput::Keyframe>& keyframes) {
    double length = 0.0;
    for (size_t i = 1; i < keyframes.size(); ++i) {
        length
            += (values.at<gtsam::Pose3>(keys::Pose(MapPoseId{keyframes[i].id}))
                    .translation()
                - values
                      .at<gtsam::Pose3>(
                          keys::Pose(MapPoseId{keyframes[i - 1].id}))
                      .translation())
                   .norm();
    }
    return length;
}

// The landmark's observations from known keyframes it lies in front of.
// None when it is behind one of them, or when its rays meet at less than
// min_parallax_rad: its depth would be all but unobservable
std::vector<std::pair<uint64_t, Eigen::Vector2d>> UsableObservations(
    const GlobalBaInput::Landmark& landmark,
    const std::map<uint64_t, const GlobalBaInput::Keyframe*>& keyframes,
    double min_depth, double min_parallax_rad) {
    std::vector<std::pair<uint64_t, Eigen::Vector2d>> usable;
    std::vector<Eigen::Vector3d> rays;
    std::set<uint64_t> seen_by;
    for (const auto& [keyframe, pixel] : landmark.observations) {
        const auto it = keyframes.find(keyframe);
        if (it == keyframes.end() || seen_by.contains(keyframe)) {
            continue;
        }
        const gtsam::Point3 in_camera
            = it->second->pose.transformTo(gtsam::Point3(landmark.position));
        if (in_camera.z() < min_depth) {
            // Behind or too close to one camera: the whole landmark is
            // suspect
            return {};
        }
        seen_by.insert(keyframe);
        usable.emplace_back(keyframe, pixel);
        rays.push_back((landmark.position - it->second->pose.translation())
                           .normalized());
    }

    // The widest angle between two rays, against the first one: enough to
    // tell a landmark seen from one place from one seen from several
    double widest = 0.0;
    for (const Eigen::Vector3d& ray : rays) {
        widest = std::max(
            widest, std::acos(std::clamp(rays.front().dot(ray), -1.0, 1.0)));
    }
    if (widest < min_parallax_rad) {
        return {};
    }
    return usable;
}

Problem BuildProblem(
    const GlobalBaInput& input, const GlobalBaOptions& options,
    const std::map<uint64_t, std::vector<std::pair<uint64_t, Eigen::Vector2d>>>&
        observations) {
    using gtsam::noiseModel::Diagonal;
    using gtsam::noiseModel::Isotropic;

    Problem problem;

    std::map<uint64_t, const GlobalBaInput::Keyframe*> by_id;
    for (const auto& keyframe : input.keyframes) {
        by_id.emplace(keyframe.id, &keyframe);
        problem.initial.insert(keys::Pose(MapPoseId{keyframe.id}),
                               keyframe.pose);
        problem.initial.insert(keys::Velocity(MapPoseId{keyframe.id}),
                               gtsam::Vector3(keyframe.velocity));
        problem.initial.insert(keys::Bias(MapPoseId{keyframe.id}),
                               keyframe.bias);
    }

    // Gauge: the first keyframe stays where the loop-corrected map has it
    const auto& first = input.keyframes.front();
    problem.graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        keys::Pose(MapPoseId{first.id}), first.pose, Isotropic::Sigma(6, 1e-4));
    const gtsam::Vector6 bias_sigmas{
        options.prior_accel_bias_sigma, options.prior_accel_bias_sigma,
        options.prior_accel_bias_sigma, options.prior_gyro_bias_sigma,
        options.prior_gyro_bias_sigma,  options.prior_gyro_bias_sigma};
    problem.graph
        .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            keys::Bias(MapPoseId{first.id}), first.bias,
            Diagonal::Sigmas(bias_sigmas));

    std::set<uint64_t> constrained_motion;
    for (const auto& edge : input.imu_edges) {
        const auto from = by_id.find(edge.from);
        if (from == by_id.end() || !by_id.contains(edge.to)) {
            continue;
        }
        gtsam::PreintegratedCombinedMeasurements combined{input.imu_params,
                                                          from->second->bias};
        wslam::map::utils::IntegrateImuWindow(combined, edge.samples,
                                              edge.from_ts, edge.to_ts);
        problem.graph.emplace_shared<gtsam::CombinedImuFactor>(
            keys::Pose(MapPoseId{edge.from}),
            keys::Velocity(MapPoseId{edge.from}),
            keys::Pose(MapPoseId{edge.to}), keys::Velocity(MapPoseId{edge.to}),
            keys::Bias(MapPoseId{edge.from}), keys::Bias(MapPoseId{edge.to}),
            combined);
        constrained_motion.insert(edge.from);
        constrained_motion.insert(edge.to);
        ++problem.imu_factors;
    }

    // A keyframe without an IMU edge (a gap in the archive) keeps its
    // velocity and bias where the smoother had them
    for (const auto& keyframe : input.keyframes) {
        if (constrained_motion.contains(keyframe.id)) {
            continue;
        }
        problem.graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            keys::Velocity(MapPoseId{keyframe.id}),
            gtsam::Vector3(keyframe.velocity), Isotropic::Sigma(3, 0.1));
        problem.graph
            .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                keys::Bias(MapPoseId{keyframe.id}), keyframe.bias,
                Diagonal::Sigmas(bias_sigmas));
    }

    const auto pixel_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(options.huber_threshold),
        Isotropic::Sigma(2, options.pixel_sigma));

    for (const auto& landmark : input.landmarks) {
        const auto it = observations.find(landmark.id);
        if (it == observations.end()
            || it->second.size() < options.min_observations) {
            continue;
        }
        const gtsam::Key key = keys::Landmark(MapLandmarkId{landmark.id});
        problem.initial.insert(key, gtsam::Point3(landmark.position));
        for (const auto& [keyframe, pixel] : it->second) {
            problem.graph.emplace_shared<ProjectionFactor>(
                gtsam::Point2(pixel), pixel_noise,
                keys::Pose(MapPoseId{keyframe}), key, input.calibration,
                /*throwCheirality=*/false,
                /*verboseCheirality=*/false);
            ++problem.projection_factors;
        }
    }

    return problem;
}

gtsam::Values Optimise(const Problem& problem, uint32_t max_iterations,
                       size_t& iterations) {
    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(static_cast<int>(max_iterations));
    params.setOrderingType("METIS");
    params.setRelativeErrorTol(1e-5);
    gtsam::LevenbergMarquardtOptimizer optimizer(problem.graph, problem.initial,
                                                 params);
    gtsam::Values result = optimizer.optimize();
    iterations += optimizer.iterations();
    return result;
}

}  // namespace

std::expected<GlobalBaResult, std::string>
wslam::v2::mapping::loop_closure::RunGlobalBa(const GlobalBaInput& input,
                                              const GlobalBaOptions& options) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();

    if (input.keyframes.size() < 2) {
        return std::unexpected("fewer than two keyframes");
    }
    if (!input.imu_params || !input.calibration) {
        return std::unexpected("no IMU parameters or camera calibration");
    }

    std::map<uint64_t, const GlobalBaInput::Keyframe*> by_id;
    for (const auto& keyframe : input.keyframes) {
        by_id.emplace(keyframe.id, &keyframe);
    }

    std::map<uint64_t, std::vector<std::pair<uint64_t, Eigen::Vector2d>>>
        observations;
    for (const auto& landmark : input.landmarks) {
        auto usable = UsableObservations(landmark, by_id, options.min_depth,
                                         options.min_parallax_rad);
        if (usable.size() >= options.min_observations) {
            observations.emplace(landmark.id, std::move(usable));
        }
    }

    GlobalBaResult result;
    gtsam::Values estimate;
    try {
        const Problem first = BuildProblem(input, options, observations);
        spdlog::info(LOG_ID
                     " {} keyframes, {} IMU factors, {} landmarks, {} "
                     "observations; initial error {:.1f}",
                     input.keyframes.size(), first.imu_factors,
                     observations.size(), first.projection_factors,
                     first.graph.error(first.initial));
        estimate = Optimise(first, options.max_iterations, result.iterations);

        // Drop the observations the first solve could not explain, then
        // solve again on the rest
        const gtsam::Cal3_S2& calibration = *input.calibration;
        for (auto it = observations.begin(); it != observations.end();) {
            const gtsam::Point3 point = estimate.at<gtsam::Point3>(
                keys::Landmark(MapLandmarkId{it->first}));
            std::erase_if(it->second, [&](const auto& observation) {
                const gtsam::PinholeCamera<gtsam::Cal3_S2> camera{
                    estimate.at<gtsam::Pose3>(
                        keys::Pose(MapPoseId{observation.first})),
                    calibration};
                const auto [projected, in_front] = camera.projectSafe(point);
                const bool outlier = !in_front
                                     || (projected - observation.second).norm()
                                            > options.outlier_px;
                result.outliers_removed += outlier ? 1 : 0;
                return outlier;
            });
            if (it->second.size() < options.min_observations) {
                it = observations.erase(it);
            } else {
                ++it;
            }
        }

        Problem second = BuildProblem(input, options, observations);
        result.imu_factors = second.imu_factors;
        result.projection_factors = second.projection_factors;
        result.initial_error = second.graph.error(second.initial);

        // Restart from the first solve, but only where the second graph has
        // the variable
        gtsam::Values warm;
        for (const auto& [key, value] : second.initial) {
            warm.insert(key, estimate.exists(key) ? estimate.at(key) : value);
        }
        second.initial = std::move(warm);
        // The first solve did most of the work
        estimate = Optimise(second, std::max(options.max_iterations / 3, 3U),
                            result.iterations);
        result.final_error = second.graph.error(estimate);

        const double initial_length = [&] {
            gtsam::Values initial;
            for (const auto& keyframe : input.keyframes) {
                initial.insert(keys::Pose(MapPoseId{keyframe.id}),
                               keyframe.pose);
            }
            return PathLength(initial, input.keyframes);
        }();
        result.path_length_ratio
            = initial_length > 0.0
                  ? PathLength(estimate, input.keyframes) / initial_length
                  : 1.0;
    } catch (const std::exception& e) {
        return std::unexpected(std::format("optimisation threw: {}", e.what()));
    }

    result.seconds
        = std::chrono::duration<double>(Clock::now() - start).count();

    spdlog::info(LOG_ID
                 " {} iterations, error {:.1f} -> {:.1f} on {} observations "
                 "({} outliers removed), path length x{:.4f}, {:.1f} s",
                 result.iterations, result.initial_error, result.final_error,
                 result.projection_factors, result.outliers_removed,
                 result.path_length_ratio, result.seconds);

    if (!(result.final_error < result.initial_error)) {
        return std::unexpected(
            std::format("error did not decrease ({:.1f} -> {:.1f})",
                        result.initial_error, result.final_error));
    }
    if (std::abs(result.path_length_ratio - 1.0)
        > options.max_path_length_change) {
        return std::unexpected(std::format(
            "path length changed by x{:.4f}, beyond the {:.0f}% guard",
            result.path_length_ratio, 100.0 * options.max_path_length_change));
    }

    for (const auto& keyframe : input.keyframes) {
        result.poses.emplace_back(
            keyframe.id,
            estimate.at<gtsam::Pose3>(keys::Pose(MapPoseId{keyframe.id})));
    }
    for (const auto& [id, kept] : observations) {
        const gtsam::Key key = keys::Landmark(MapLandmarkId{id});
        if (estimate.exists(key)) {
            result.landmarks.emplace_back(id, estimate.at<gtsam::Point3>(key));
        }
    }
    return result;
}
