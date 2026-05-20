#include "sensor_loader.hpp"

#include <algorithm>

#include "common.hpp"
#include "compute.hpp"
#include "provider_base.hpp"

using namespace wslam;

#define LOG_ID "[Sensor Loader Pass]"

std::string SensorLoaderPass::getId() const { return LOG_ID; }

std::optional<std::string> SensorLoaderPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (!storage_.has(ResourceIdentifier::GetImuVecName())) {
        storage_.set(ResourceIdentifier::GetImuVecName(),
                     std::vector<data::IMUReading>{});
    }

    return std::nullopt;
}

std::optional<std::string> SensorLoaderPass::execute() {
    spdlog::info(LOG_ID " executing");

    if (iter_ == std::nullopt) {
        iter_ = generator_.begin();
        if (auto err = skipInitialFrames(kInitialFramesToSkip)) {
            return err;
        }
    } else if (iter_.value() == generator_.end()) {
        spdlog::info(LOG_ID " No more sensor readings. Stopping execution");
        return compute::kComputeStopExecution;
    } else {
        iter_.value()++;
    }

    std::vector<data::IMUReading> temp_imu{};

    const int kMaxIterations = 1000000;
    int counter = 0;

    ReadingType reading;
    for (; iter_ != generator_.end() && counter < kMaxIterations;
         (*iter_)++, counter++) {
        assert(iter_);

        reading = std::move(*iter_.value());

        if (!reading) {
            return "getting reading: " + reading.error();
        }

        if (reading->imu.has_value()) {
            temp_imu.emplace_back(reading->imu.value());
        }

        const auto is_empty
            = [](const auto& frame) { return !frame.has_value(); };

        // If there is a frame, that means we're done with collecting imu
        // readings
        if (std::ranges::none_of(reading->frames, is_empty)) {
            break;
        }
    }

    if (counter >= kMaxIterations) {
        spdlog::error(
            LOG_ID " Max numbre of iterations reached. cancelling execution");
        return "too many readings without a frame";
    }

    spdlog::debug(LOG_ID
                  " found frame for inserting. temporary imu readings:{} "
                  "imu:last_ts:{} frame0:ts:{}",
                  temp_imu.size(), temp_imu.back().timestamp,
                  reading->frames.front()->timestamp);

    // SET IMU DATA
    auto imu_vec = storage_.get<std::vector<data::IMUReading>>(
        ResourceIdentifier::GetImuVecName());

    if (!imu_vec.has_value()) {
        imu_vec = std::vector<data::IMUReading>{};
    }

    std::ranges::move(temp_imu, std::back_inserter(imu_vec.value()));

    storage_.set(ResourceIdentifier::GetImuVecName(), imu_vec.value());

    // SET FRAMES DATA
    uint64_t latest_ts = 0;
    for (size_t i = 0; i < reading->frames.size(); i++) {
        const auto& frame = reading->frames[i];

        if (!frame.has_value()) {
            continue;
        }

        const auto name
            = ResourceIdentifier::GetFrameName(static_cast<uint32_t>(i));

        spdlog::debug(LOG_ID " Added new frame. name:'{}', size:{}x{}", name,
                      frame->width, frame->height);
        latest_ts = frame->timestamp;
        storage_.set(name, std::move(frame).value());
    }

    // Publish the current frame's timestamp so mapping-stage passes can
    // window the IMU buffer to the [prev_kf, this_frame] interval.
    storage_.set(ResourceIdentifier::FrameTimestampNsName, latest_ts);

    return std::nullopt;
}

std::optional<std::string> SensorLoaderPass::skipInitialFrames(uint32_t count) {
    if (count == 0) {
        return std::nullopt;
    }

    spdlog::info(
        LOG_ID
        " Skipping the first {} frames (cold-start stale frames discarded)",
        count);

    uint32_t skipped = 0;
    while (skipped < count && iter_.value() != generator_.end()) {
        ReadingType reading = std::move(*iter_.value());

        if (!reading) {
            return "skipping initial frames: " + reading.error();
        }

        const auto is_empty
            = [](const auto& frame) { return !frame.has_value(); };

        // Only frames count toward the skip budget; lone IMU samples between
        // frames are discarded silently along with them.
        if (std::ranges::none_of(reading->frames, is_empty)) {
            skipped++;
        }

        (*iter_)++;
    }

    if (skipped < count) {
        spdlog::warn(LOG_ID
                     " Reached end of sensor stream while skipping initial "
                     "frames ({} skipped out of {} requested)",
                     skipped, count);
        return compute::kComputeStopExecution;
    }

    spdlog::debug(LOG_ID " Done skipping {} initial frames", skipped);

    return std::nullopt;
}
