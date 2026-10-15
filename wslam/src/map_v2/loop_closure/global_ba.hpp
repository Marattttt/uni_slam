#pragma once

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "data/provider_base.hpp"

// Export-time global visual-inertial bundle adjustment of the whole map,
// started from the loop-corrected estimate (pose graph merge)
namespace wslam::v2::mapping::loop_closure {

struct GlobalBaInput {
    struct Keyframe {
        uint64_t id = 0;
        // Initial guess: loop-corrected pose, its velocity rotated with it
        gtsam::Pose3 pose;
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        // The smoother's estimate: the IMU edge leaving this keyframe is
        // preintegrated with it, and the bias variable starts there
        gtsam::imuBias::ConstantBias bias;
    };

    struct ImuEdge {
        uint64_t from = 0;
        uint64_t to = 0;
        uint64_t from_ts = 0;
        uint64_t to_ts = 0;
        std::vector<data::IMUReading> samples;
    };

    struct Landmark {
        uint64_t id = 0;
        // Initial guess, loop-corrected
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        // Keyframe and undistorted LOD-0 pixel of every observation
        std::vector<std::pair<uint64_t, Eigen::Vector2d>> observations;
    };

    // In id order; the first one is the gauge
    std::vector<Keyframe> keyframes;
    std::vector<ImuEdge> imu_edges;
    std::vector<Landmark> landmarks;

    // Gravity in the world frame and the camera/IMU extrinsics
    std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params;
    // Pinhole camera of the undistorted observations
    std::shared_ptr<gtsam::Cal3_S2> calibration;
};

struct GlobalBaOptions {
    // Observation noise and the Huber threshold, in sigmas
    double pixel_sigma = 1.5;
    double huber_threshold = 1.345;
    // Landmarks seen by fewer keyframes are left out
    size_t min_observations = 3;
    // A landmark must be at least this far in front of every camera seeing it
    double min_depth = 0.1;
    // ... and seen along rays at least this far apart
    double min_parallax_rad = 1.0 * std::numbers::pi / 180.0;
    // After the first solve, observations reprojecting further than this are
    // dropped and the map is solved again
    double outlier_px = 4.0;
    uint32_t max_iterations = 15;

    // Prior on the first keyframe's bias, centred on the smoother's estimate
    double prior_accel_bias_sigma = 0.1;
    double prior_gyro_bias_sigma = 0.01;

    // The result is rejected when it changes the trajectory's length by more
    // than this fraction: a guard against scale collapse
    double max_path_length_change = 0.05;
};

struct GlobalBaResult {
    std::vector<std::pair<uint64_t, gtsam::Pose3>> poses;
    std::vector<std::pair<uint64_t, Eigen::Vector3d>> landmarks;

    size_t imu_factors = 0;
    size_t projection_factors = 0;
    size_t outliers_removed = 0;
    size_t iterations = 0;
    // Graph error of the initial guess and of the result, both measured on
    // the final (outlier-free) graph
    double initial_error = 0.0;
    double final_error = 0.0;
    // Trajectory length, result / initial guess
    double path_length_ratio = 1.0;
    double seconds = 0.0;
};

// Fails, with the reason, when the input is unusable, the optimisation
// throws, or the result fails the safety checks; the caller then keeps the
// initial guess
[[nodiscard]] std::expected<GlobalBaResult, std::string> RunGlobalBa(
    const GlobalBaInput& input, const GlobalBaOptions& options);

}  // namespace wslam::v2::mapping::loop_closure
