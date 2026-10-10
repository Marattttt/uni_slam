#include "sensor_loader.hpp"

#include <algorithm>
#include <utility>

#include "assert.hpp"
#include "common.hpp"
#include "compute/compute.hpp"
#include "data/provider_base.hpp"

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

    if (stopped_) {
        return compute::kComputeStopExecution;
    }

    if (!iter_.has_value()) {
        iter_ = generator_.begin();
        if (auto err = skipInitialFrames(kInitialFramesToSkip)) {
            return err;
        }
    } else if (*iter_ == generator_.end()) {
        return stop("No more sensor readings");
    } else {
        // Still points at the reading the previous execute() published
        ++*iter_;
    }

    std::vector<data::IMUReading> imu;
    auto reading = findNextFrame(imu);
    if (!reading) {
        // An error, or the stop sentinel when the stream ended
        return std::move(reading).error();
    }

    return publish(std::move(reading).value(), std::move(imu));
}

std::expected<data::Reading<1>, std::string> SensorLoaderPass::findNextFrame(
    std::vector<data::IMUReading>& imu) {
    using Kind = data::ReadingCandidate::Kind;
    using Decision = data::ReadingDecision;

    WSLAM_ASSERT(iter_.has_value(), "searching before the stream was started");
    auto& iter = *iter_;

    const auto is_empty = [](const auto& frame) { return !frame.has_value(); };

    const int kMaxIterations = 1000000;
    for (int counter = 0; iter != generator_.end(); ++iter, ++counter) {
        if (counter >= kMaxIterations) {
            spdlog::error(LOG_ID
                          " Max numbre of iterations reached. cancelling "
                          "execution");
            return std::unexpected("too many readings without a frame");
        }

        ReadingType reading = std::move(*iter);
        if (!reading) {
            return std::unexpected("getting reading: " + reading.error());
        }

        if (const auto& sample = reading->imu) {
            auto decision = filter(Kind::Imu, sample->timestamp);
            if (!decision) {
                return std::unexpected("filtering imu reading: "
                                       + std::move(decision).error());
            }
            if (decision.value() == Decision::StopStream) {
                return std::unexpected(
                    stop("Reading filter ended the stream at an imu reading"));
            }
            if (decision.value() == Decision::Keep) {
                imu.push_back(*sample);
            }
        }

        if (std::ranges::any_of(reading->frames, is_empty)) {
            continue;
        }

        // Every camera's frame of a reading shares one timestamp
        auto decision = filter(Kind::Frame, reading->frames.front()->timestamp);
        if (!decision) {
            return std::unexpected("filtering frame: "
                                   + std::move(decision).error());
        }
        if (decision.value() == Decision::StopStream) {
            return std::unexpected(
                stop("Reading filter ended the stream at a frame"));
        }
        // A kept frame ends the search; the iterator stays on its reading
        if (decision.value() == Decision::Keep) {
            return std::move(reading).value();
        }
    }

    return std::unexpected(stop("No more sensor readings"));
}

std::optional<std::string> SensorLoaderPass::publish(
    data::Reading<1> reading, std::vector<data::IMUReading> imu) {
    spdlog::debug(LOG_ID
                  " found frame for inserting. temporary imu readings:{}",
                  imu.size());

    // SET IMU DATA
    auto imu_vec = storage_.get<std::vector<data::IMUReading>>(
        ResourceIdentifier::GetImuVecName());

    if (!imu_vec.has_value()) {
        imu_vec = std::vector<data::IMUReading>{};
    }

    std::ranges::move(imu, std::back_inserter(imu_vec.value()));

    storage_.set(ResourceIdentifier::GetImuVecName(), imu_vec.value());

    // SET FRAMES DATA
    uint64_t latest_ts = 0;
    for (size_t i = 0; i < reading.frames.size(); i++) {
        const auto& pending = reading.frames[i];
        WSLAM_ASSERT(pending.has_value(), "kept frame {} has no camera {}",
                     frame_count_ - 1, i);

        auto frame = pending->load();
        if (!frame) {
            return "loading frame: " + std::move(frame).error();
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

std::expected<data::ReadingDecision, std::string> SensorLoaderPass::filter(
    data::ReadingCandidate::Kind kind, uint64_t timestamp) {
    uint64_t& count = kind == data::ReadingCandidate::Kind::Frame ? frame_count_
                                                                  : imu_count_;
    const data::ReadingCandidate candidate{
        .kind = kind,
        .timestamp = timestamp,
        .index = count++,
    };

    if (!filter_) {
        return data::ReadingDecision::Keep;
    }

    return filter_(candidate);
}

std::string SensorLoaderPass::stop(std::string_view reason) {
    spdlog::info(LOG_ID " {}. Stopping execution", reason);
    stopped_ = true;
    return compute::kComputeStopExecution;
}

std::optional<std::string> SensorLoaderPass::skipInitialFrames(uint32_t count) {
    if (count == 0) {
        return std::nullopt;
    }

    WSLAM_ASSERT(iter_.has_value(), "skipping before the stream was started");

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

        // Skipped items still count toward the indices the filter sees
        if (reading->imu.has_value()) {
            imu_count_++;
        }

        const auto is_empty
            = [](const auto& frame) { return !frame.has_value(); };

        // Only frames count toward the skip budget; lone IMU samples between
        // frames are discarded silently along with them.
        if (std::ranges::none_of(reading->frames, is_empty)) {
            frame_count_++;
            skipped++;
        }

        (*iter_)++;
    }

    if (skipped < count) {
        spdlog::warn(LOG_ID
                     " Reached end of sensor stream while skipping initial "
                     "frames ({} skipped out of {} requested)",
                     skipped, count);
        stopped_ = true;
        return compute::kComputeStopExecution;
    }

    spdlog::debug(LOG_ID " Done skipping {} initial frames", skipped);

    return std::nullopt;
}
