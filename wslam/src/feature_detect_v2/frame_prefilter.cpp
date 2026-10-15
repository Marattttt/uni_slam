#include "frame_prefilter.hpp"

#include <spdlog/spdlog.h>

#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "compute/stage.hpp"
#include "map/utils.hpp"
#include "models.hpp"
#include "pose_estimate_v2/util/camera.hpp"

using namespace wslam;
namespace utils = wslam::map::utils;

#define LOG_ID "[Frame Prefilter pass]"

v2::FramePrefilterOptions v2::ReadFramePrefilterOptionsFromEnv() {
    FramePrefilterOptions options;
    MotionPredictionOptions& prediction = options.prediction;

    prediction.confidence_sigmas = utils::EnvDouble(
        "WSLAM_PREFILTER_SIGMAS", prediction.confidence_sigmas);
    prediction.gyro_bias_sigma = utils::EnvDouble(
        "WSLAM_PREFILTER_GYRO_BIAS_SIGMA", prediction.gyro_bias_sigma);
    prediction.accel_bias_sigma = utils::EnvDouble(
        "WSLAM_PREFILTER_ACCEL_BIAS_SIGMA", prediction.accel_bias_sigma);
    prediction.attitude_sigma = utils::EnvDouble(
        "WSLAM_PREFILTER_ATTITUDE_SIGMA", prediction.attitude_sigma);
    prediction.velocity_sigma = utils::EnvDouble(
        "WSLAM_PREFILTER_VELOCITY_SIGMA", prediction.velocity_sigma);
    prediction.visual_rotation_sigma
        = utils::EnvDouble("WSLAM_PREFILTER_VISUAL_ROTATION_SIGMA",
                           prediction.visual_rotation_sigma);
    prediction.depth_sigma = utils::EnvDouble("WSLAM_PREFILTER_DEPTH_SIGMA",
                                              prediction.depth_sigma);
    options.max_consecutive_skips = utils::EnvU32(
        "WSLAM_PREFILTER_MAX_SKIPS", options.max_consecutive_skips);

    spdlog::info(LOG_ID
                 " {} sigmas; gyro bias {} rad/s, accel bias {} m/s^2, "
                 "attitude {} rad, velocity >= {} m/s, depth {}; at most {} "
                 "skips in a row",
                 prediction.confidence_sigmas, prediction.gyro_bias_sigma,
                 prediction.accel_bias_sigma, prediction.attitude_sigma,
                 prediction.velocity_sigma, prediction.depth_sigma,
                 options.max_consecutive_skips);

    return options;
}

std::string v2::FramePrefilterPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::FramePrefilterPass::initialize() {
    spdlog::info(LOG_ID " Initializing");

    auto camera = util::ReadCameraParams(storage_, options_.camera_index);
    if (!camera) {
        return std::move(camera).error();
    }
    auto imu_params = storage_.get<data::IMUSensorParams>(
        ResourceIdentifier::ImuParamsName);
    if (!imu_params) {
        return "no IMU parameters";
    }
    predictor_.emplace(std::move(camera).value(), std::move(imu_params).value(),
                       options_.prediction);

    return std::nullopt;
}

std::optional<std::string> v2::FramePrefilterPass::execute() {
    WSLAM_ASSERT(predictor_.has_value(), "executing before initialize()");
    ++frames_;

    const auto prior = storage_.getPtr<KeyframeMotionPrior>(
        ResourceIdentifier::KeyframeMotionPriorName);
    if (!prior) {
        // Before the second keyframe, or after one without a usable depth
        consecutive_skips_ = 0;
        return std::nullopt;
    }

    const auto frame_ts
        = storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName);
    if (!frame_ts) {
        return "no frame timestamp";
    }
    const auto imu = storage_.getPtr<std::vector<data::IMUReading>>(
        ResourceIdentifier::GetImuVecName());
    if (!imu) {
        return "no IMU buffer";
    }

    auto prediction
        = predictor_->predict(*prior.value(), *imu.value(), frame_ts.value());
    if (!prediction) {
        spdlog::warn(LOG_ID " No prediction ({}); running the model",
                     prediction.error());
        consecutive_skips_ = 0;
        return std::nullopt;
    }

    if (prediction->small_motion
        && consecutive_skips_ < options_.max_consecutive_skips) {
        ++consecutive_skips_;
        ++skipped_;
        spdlog::info(LOG_ID
                     " Skipping: {:.3f} s after the keyframe, rotation "
                     "{:.4f} (<= {:.4f}) rad, parallax {:.2f} (<= {:.2f}) px; "
                     "{} in a row, {} of {} frames",
                     prediction->dt_s, prediction->rotation_rad,
                     prediction->rotation_upper_rad, prediction->parallax_px,
                     prediction->parallax_upper_px, consecutive_skips_,
                     skipped_, frames_);
        storage_.set(ResourceIdentifier::FrameSkippedName, true);
        return compute::kStageStopExecution;
    }

    spdlog::info(LOG_ID
                 " Running the model: {:.3f} s after the keyframe, rotation "
                 "{:.4f} (<= {:.4f}) rad, parallax {:.2f} (<= {:.2f}) px{}",
                 prediction->dt_s, prediction->rotation_rad,
                 prediction->rotation_upper_rad, prediction->parallax_px,
                 prediction->parallax_upper_px,
                 prediction->small_motion ? " (skip cap reached)" : "");
    consecutive_skips_ = 0;
    storage_.set(ResourceIdentifier::KeyframeMotionPredictionName,
                 std::move(prediction).value());

    return std::nullopt;
}
