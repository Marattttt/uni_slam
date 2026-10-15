#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

#include "data/provider_base.hpp"
#include "models.hpp"

// Prediction of the keyframe gate's small-motion test from the IMU: how far
// the camera rotated and how far the image moved since the last keyframe, with
// upper bounds that account for the IMU's noise and the uncertainty of the
// keyframe's state
namespace wslam::v2 {

struct MotionPredictionOptions {
    // Width of the upper bounds, in standard deviations
    double confidence_sigmas = 3.0;

    // 1-sigma errors of the keyframe state the prior carries. The IMU's own
    // noise densities and random walks come from its sensor parameters
    double gyro_bias_sigma = 0.005;  // rad/s
    double accel_bias_sigma = 0.1;   // m/s^2
    double attitude_sigma = 0.01;    // rad, gravity direction
    // Floor of the prior's velocity sigma (which is 0 until measured)
    double velocity_sigma = 0.05;  // m/s
    // Noise of the rotation the gate measures from the images
    double visual_rotation_sigma = 0.002;  // rad
    // Relative error of the median scene depth
    double depth_sigma = 0.15;

    // Image grid the median parallax is predicted over
    uint32_t grid_columns = 8;
    uint32_t grid_rows = 6;
    // Grid inset from the image border, as a fraction of its size
    double grid_margin = 0.05;
};

// Predicts the motion from a prior's keyframe to a later frame by
// preintegrating the IMU samples after the keyframe, and the median parallax
// that motion causes for a scene at the keyframe's median depth.
// small_motion is set when both upper bounds are below the gate's thresholds.
// Keeps the image grid and the IMU calibration (rebuilt when the prior's
// gravity changes) between calls
class MotionPredictor {
   public:
    MotionPredictor(data::CamSensorParams camera,
                    data::IMUSensorParams imu_params,
                    MotionPredictionOptions options);
    ~MotionPredictor();

    MotionPredictor(const MotionPredictor&) = delete;
    MotionPredictor& operator=(const MotionPredictor&) = delete;
    MotionPredictor(MotionPredictor&&) noexcept;
    MotionPredictor& operator=(MotionPredictor&&) noexcept;

    [[nodiscard]] std::expected<KeyframeMotionPrediction, std::string> predict(
        const KeyframeMotionPrior& prior, std::span<const data::IMUReading> imu,
        uint64_t frame_ts);

    [[nodiscard]] const MotionPredictionOptions& options() const {
        return options_;
    }

   private:
    struct State;

    data::CamSensorParams camera_;
    data::IMUSensorParams imu_params_;
    MotionPredictionOptions options_;
    std::unique_ptr<State> state_;
};

}  // namespace wslam::v2
