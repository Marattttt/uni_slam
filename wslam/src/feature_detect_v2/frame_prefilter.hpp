#pragma once

#include <cstdint>
#include <optional>
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
    uint32_t camera_index = 0;
};

// Defaults overridden by the WSLAM_PREFILTER_* env variables: _SIGMAS,
// _GYRO_BIAS_SIGMA, _ACCEL_BIAS_SIGMA, _ATTITUDE_SIGMA, _VELOCITY_SIGMA,
// _VISUAL_ROTATION_SIGMA, _DEPTH_SIGMA, _MAX_SKIPS
[[nodiscard]] FramePrefilterOptions ReadFramePrefilterOptionsFromEnv();

// Skips the model on frames whose motion since the last keyframe the IMU
// predicts, with confidence, to be too small for the keyframe gate. Runs
// before any GPU work of the feature detect stage; stopping the stage leaves
// the reference marker and the cached features for the next frame that runs.
//
// Input  (from AnyBag): KeyframeMotionPrior under KeyframeMotionPriorName
//                       (none before the mapping stage published one: every
//                       frame runs), IMU buffer, frame timestamp, sensor
//                       parameters
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

    uint32_t consecutive_skips_ = 0;
    uint64_t frames_ = 0;
    uint64_t skipped_ = 0;
};

}  // namespace wslam::v2
