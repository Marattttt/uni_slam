#include "motion_prediction.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <utility>
#include <vector>

#include "map/map_helpers_lods.hpp"
#include "map/utils.hpp"

using namespace wslam;

namespace {
// Integration noise on top of the per-sample covariance; irrelevant to the
// mean the prediction uses
constexpr double kIntegrationSigma = 1e-4;

constexpr uint32_t kUndistortIterations = 20;
constexpr double kUndistortTolerance = 1e-9;

// A point this close to (or behind) the current camera after the predicted
// motion has no meaningful parallax: the frame is not predicted small
constexpr double kMinProjectedDepth = 1e-3;

struct GridGeometry {
    // Undistorted rays of the grid pixels (normalised image plane)
    std::vector<Eigen::Vector2d> rays;
    // Largest distance of a grid ray from the principal point, in pixels
    double max_radius_px = 0.0;
};

GridGeometry MakeGrid(const data::CamSensorParams& camera,
                      const v2::MotionPredictionOptions& options) {
    const auto [width, height] = camera.resolution;
    const Eigen::Vector4d& k = camera.intrinsics;  // fu, fv, cu, cv

    GridGeometry grid;
    grid.rays.reserve(size_t{options.grid_columns} * options.grid_rows);

    const auto spread = [&](uint32_t i, uint32_t count, int size) {
        const double margin = options.grid_margin * size;
        const double span = size - 1 - 2 * margin;
        return count <= 1 ? (size - 1) / 2.0
                          : margin + span * i / static_cast<double>(count - 1);
    };

    for (uint32_t row = 0; row < options.grid_rows; ++row) {
        for (uint32_t column = 0; column < options.grid_columns; ++column) {
            const Eigen::Vector2d pixel{
                spread(column, options.grid_columns, width),
                spread(row, options.grid_rows, height)};
            const Eigen::Vector2d ray = map::util::PixelToNormalised(
                pixel, camera.intrinsics, camera.distortion_coefficients,
                kUndistortIterations, kUndistortTolerance);
            grid.rays.push_back(ray);
            grid.max_radius_px = std::max(
                grid.max_radius_px,
                Eigen::Vector2d{k[0] * ray.x(), k[1] * ray.y()}.norm());
        }
    }

    return grid;
}

// Median pinhole pixel displacement of the grid's points at depth (along the
// optical axis) under p_curr = rotation * p_prev + translation. Infinite when
// a point leaves the front of the camera
double MedianParallax(const GridGeometry& grid, const Eigen::Matrix3d& rotation,
                      const Eigen::Vector3d& translation, double depth,
                      const Eigen::Vector4d& intrinsics) {
    const double fu = intrinsics[0];
    const double fv = intrinsics[1];

    std::vector<double> parallaxes;
    parallaxes.reserve(grid.rays.size());
    for (const Eigen::Vector2d& ray : grid.rays) {
        const Eigen::Vector3d point = depth * ray.homogeneous();
        const Eigen::Vector3d moved = rotation * point + translation;
        if (moved.z() < kMinProjectedDepth) {
            parallaxes.push_back(std::numeric_limits<double>::infinity());
            continue;
        }
        const Eigen::Vector2d shift = moved.hnormalized() - ray;
        parallaxes.push_back(
            Eigen::Vector2d{fu * shift.x(), fv * shift.y()}.norm());
    }

    const auto middle = parallaxes.begin()
                        + static_cast<std::ptrdiff_t>(parallaxes.size() / 2);
    std::nth_element(parallaxes.begin(), middle, parallaxes.end());
    return *middle;
}

// Unit directions the motion errors are sampled along: the 6 axis directions
// and the 8 diagonals, so a worst-case error direction is never more than
// ~35 degrees from a sample
std::vector<Eigen::Vector3d> MakeErrorDirections() {
    std::vector<Eigen::Vector3d> directions;
    for (int axis = 0; axis < 3; ++axis) {
        for (const double sign : {-1.0, 1.0}) {
            Eigen::Vector3d direction = Eigen::Vector3d::Zero();
            direction[axis] = sign;
            directions.push_back(direction);
        }
    }
    for (const double x : {-1.0, 1.0}) {
        for (const double y : {-1.0, 1.0}) {
            for (const double z : {-1.0, 1.0}) {
                directions.push_back(Eigen::Vector3d{x, y, z}.normalized());
            }
        }
    }
    return directions;
}
}  // namespace

struct v2::MotionPredictor::State {
    GridGeometry grid;
    std::vector<Eigen::Vector3d> error_directions;
    std::shared_ptr<gtsam::PreintegrationCombinedParams> calibration;
    Eigen::Vector3d calibration_gravity = Eigen::Vector3d::Zero();
};

v2::MotionPredictor::MotionPredictor(data::CamSensorParams camera,
                                     data::IMUSensorParams imu_params,
                                     MotionPredictionOptions options)
    : camera_(std::move(camera)),
      imu_params_(std::move(imu_params)),
      options_(options),
      state_(std::make_unique<State>()) {
    state_->grid = MakeGrid(camera_, options_);
    state_->error_directions = MakeErrorDirections();
}

v2::MotionPredictor::~MotionPredictor() = default;
v2::MotionPredictor::MotionPredictor(MotionPredictor&&) noexcept = default;
v2::MotionPredictor& v2::MotionPredictor::operator=(MotionPredictor&&) noexcept
    = default;

std::expected<v2::KeyframeMotionPrediction, std::string>
v2::MotionPredictor::predict(const KeyframeMotionPrior& prior,
                             std::span<const data::IMUReading> imu,
                             uint64_t frame_ts) {
    const data::CamSensorParams& camera = camera_;
    const data::IMUSensorParams& imu_params = imu_params_;
    const MotionPredictionOptions& options = options_;

    if (frame_ts < prior.keyframe_ts) {
        return std::unexpected(
            std::format("frame at {} ns precedes the keyframe at {} ns",
                        frame_ts, prior.keyframe_ts));
    }
    if (!(prior.median_depth_m > 0.0)) {
        return std::unexpected("the prior has no scene depth");
    }
    if (options.grid_columns == 0 || options.grid_rows == 0) {
        return std::unexpected("empty parallax grid");
    }

    const double dt = static_cast<double>(frame_ts - prior.keyframe_ts) * 1e-9;

    // Samples after the frame belong to later frames
    const auto past_frame = std::ranges::find_if(
        imu,
        [frame_ts](const auto& sample) { return sample.timestamp > frame_ts; });
    imu = imu.first(static_cast<size_t>(past_frame - imu.begin()));

    // Mean motion: the mapping stage's own preintegration, from the
    // keyframe's state. The pose variables are camera poses, so the predicted
    // NavState is the camera's
    const gtsam::imuBias::ConstantBias bias{prior.accel_bias, prior.gyro_bias};
    if (!state_->calibration
        || state_->calibration_gravity != prior.gravity_world) {
        state_->calibration = map::utils::CreateImuCalibration(
            camera, imu_params, prior.gravity_world, kIntegrationSigma);
        state_->calibration_gravity = prior.gravity_world;
    }
    gtsam::PreintegratedCombinedMeasurements preintegrated{state_->calibration,
                                                           bias};
    map::utils::IntegrateImuWindow(preintegrated, imu, prior.keyframe_ts,
                                   frame_ts);

    const gtsam::NavState keyframe{gtsam::Rot3{prior.R_world_cam},
                                   gtsam::Point3::Zero(), prior.velocity_world};
    const gtsam::NavState current = preintegrated.predict(keyframe, bias);

    // p_curr = R * p_prev + t between the camera frames, as the gate's
    // pose estimate reports it
    const Eigen::Matrix3d r_wc_prev = prior.R_world_cam;
    const Eigen::Matrix3d r_wc_curr = current.pose().rotation().matrix();
    const Eigen::Matrix3d rotation = r_wc_curr.transpose() * r_wc_prev;
    const Eigen::Vector3d translation
        = r_wc_curr.transpose() * (keyframe.position() - current.position());

    // Uncertainty of the angle: gyroscope bias error, angle random walk, bias
    // random walk
    const double gyro_noise = imu_params.gyroscope_noise_density;
    const double gyro_walk = imu_params.gyroscope_random_walk;
    const double rotation_sigma
        = std::sqrt(options.gyro_bias_sigma * options.gyro_bias_sigma * dt * dt
                    + gyro_noise * gyro_noise * dt
                    + gyro_walk * gyro_walk * dt * dt * dt / 3.0);

    // Uncertainty of the translation: velocity error, accelerometer bias and
    // gravity-direction error, velocity random walk, bias random walk
    const double velocity_sigma
        = std::max(options.velocity_sigma, prior.velocity_sigma);
    const double tilt_acceleration
        = prior.gravity_world.norm() * options.attitude_sigma;
    const double accel_noise = imu_params.accelerometer_noise_density;
    const double accel_walk = imu_params.accelerometer_random_walk;
    const double dt2 = dt * dt;
    const double translation_sigma
        = std::sqrt(velocity_sigma * velocity_sigma * dt2
                    + (options.accel_bias_sigma * options.accel_bias_sigma
                       + tilt_acceleration * tilt_acceleration)
                          * dt2 * dt2 / 4.0
                    + accel_noise * accel_noise * dt2 * dt / 3.0
                    + accel_walk * accel_walk * dt2 * dt2 * dt / 20.0);

    const GridGeometry& grid = state_->grid;
    const double k = options.confidence_sigmas;

    KeyframeMotionPrediction prediction;
    prediction.dt_s = dt;
    prediction.rotation_rad = Eigen::AngleAxisd{rotation}.angle();
    // The gate compares its own, visual, rotation estimate
    prediction.rotation_upper_rad
        = prediction.rotation_rad
          + k * std::hypot(rotation_sigma, options.visual_rotation_sigma);

    prediction.parallax_px = MedianParallax(
        grid, rotation, translation, prior.median_depth_m, camera.intrinsics);

    // Upper bound: the largest median parallax over the motions the errors
    // allow, each error at k sigmas along one of the sampled directions, for
    // the nominal scene depth and one as close as the depth's error allows.
    // The median is taken per motion, as the gate does: bounding each grid
    // point by its own worst case would add the image-corner error to the
    // median (2-3x looser on V1_01_easy)
    const double near_depth
        = prior.median_depth_m / (1.0 + k * options.depth_sigma);
    const double rotation_error = k * rotation_sigma;
    const double translation_error = k * translation_sigma;
    const std::vector<Eigen::Vector3d>& directions = state_->error_directions;

    double parallax_upper = prediction.parallax_px;
    for (size_t r = 0; r <= directions.size(); ++r) {
        const Eigen::Matrix3d rotated
            = r == 0 ? rotation
                     : Eigen::Matrix3d{
                           Eigen::AngleAxisd{rotation_error,
                                             directions[r - 1]}
                           * rotation};
        for (size_t t = 0; t <= directions.size(); ++t) {
            const Eigen::Vector3d moved
                = t == 0 ? translation
                         : Eigen::Vector3d{translation
                                           + translation_error
                                                 * directions[t - 1]};
            for (const double depth : {prior.median_depth_m, near_depth}) {
                parallax_upper = std::max(
                    parallax_upper, MedianParallax(grid, rotated, moved, depth,
                                                   camera.intrinsics));
            }
        }
    }
    prediction.parallax_upper_px = parallax_upper;

    prediction.small_motion
        = prediction.rotation_upper_rad < prior.min_rotation_rad
          && prediction.parallax_upper_px < prior.min_parallax_px;

    return prediction;
}
