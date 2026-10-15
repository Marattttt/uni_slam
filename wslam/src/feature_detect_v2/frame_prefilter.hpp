#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "motion_prediction.hpp"

namespace wslam::v2 {

struct FramePrefilterOptions {
    MotionPredictionOptions prediction;
    // After this many skipped frames in a row the model runs anyway, so a
    // wrong prediction can't hold the keyframe back for long
    uint32_t max_consecutive_skips = 10;

    // Before there is a motion prior (no keyframe yet, so the IMU can't
    // predict the gate): skip the frames whose last stationary_window_samples
    // IMU samples score below stationary_max_score (map::utils::
    // StationaryScore; parked on V1_01_easy 0.13-0.29, rotor spin-up and
    // flight >= 0.9). The score can't see constant rotation or constant
    // velocity, so the model still runs after max_stationary_skips in a row
    uint32_t stationary_window_samples = 50;
    double stationary_max_score = 0.5;
    uint32_t max_stationary_skips = 40;

    uint32_t camera_index = 0;
};

// Defaults overridden by the WSLAM_PREFILTER_* env variables: _SIGMAS,
// _GYRO_BIAS_SIGMA, _ACCEL_BIAS_SIGMA, _ATTITUDE_SIGMA, _VELOCITY_SIGMA,
// _VISUAL_ROTATION_SIGMA, _DEPTH_SIGMA, _MAX_SKIPS, _STATIONARY_WINDOW,
// _STATIONARY_SCORE, _MAX_STATIONARY_SKIPS
[[nodiscard]] FramePrefilterOptions ReadFramePrefilterOptionsFromEnv();

// Skips the model on frames whose motion since the last keyframe the IMU
// predicts, with confidence, to be too small for the keyframe gate, and,
// before the mapping stage published a motion prior, on frames the IMU sees
// stationary (the dataset's parked lead-in). Runs before any GPU work of the
// feature detect stage; stopping the stage leaves the reference marker and the
// cached features for the next frame that runs.
//
// Input  (from AnyBag): KeyframeMotionPrior under KeyframeMotionPriorName
//                       (none before the mapping stage published one: only
//                       stationary frames are skipped), IMU buffer, frame
//                       timestamp, sensor parameters
// Output (to   AnyBag): FrameSkippedName marker on skipped frames,
//                       KeyframeMotionPrediction under
//                       KeyframeMotionPredictionName on the others
class FramePrefilterPass : public compute::Pass {
   public:
    FramePrefilterPass(AnyBag& storage, FramePrefilterOptions options)
        : storage_(storage), options_(options) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    const FramePrefilterOptions options_;

    // Set by initialize()
    std::optional<MotionPredictor> predictor_;

    // Stationary-skip decision for a frame without a motion prior
    [[nodiscard]] bool isStationary(std::span<const data::IMUReading> imu,
                                    uint64_t frame_ts, double& score) const;

    uint32_t consecutive_skips_ = 0;
    uint32_t stationary_skips_ = 0;
    uint64_t frames_ = 0;
    uint64_t skipped_ = 0;
    uint64_t stationary_skipped_ = 0;
    // Frames the pass let through to the model
    uint64_t model_runs_ = 0;
};

}  // namespace wslam::v2
