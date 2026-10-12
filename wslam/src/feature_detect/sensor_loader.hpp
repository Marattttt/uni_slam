#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"

namespace wslam {
// Pulls readings until it finds the next frame, publishes that frame, its
// timestamp and the IMU samples read on the way. An optional filter sees
// every frame and IMU sample before any file behind it is read, and decides
// whether it is kept, skipped, or ends the stream
class SensorLoaderPass : public compute::Pass {
   public:
    using ReadingType = std::expected<data::Reading<1>, std::string>;

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    SensorLoaderPass(std::generator<ReadingType>&& generator, AnyBag& storage,
                     data::ReadingFilter filter = {})
        : storage_(storage),
          generator_(std::move(generator)),
          filter_(std::move(filter)) {};

   private:
    // Number of initial frames to drop on the first execute(). Zero: the
    // mapping stage *needs* the dataset's stationary lead-in — gravity and
    // gyro-bias are initialised from a stationary IMU window and the first
    // keyframe's velocity prior assumes ~zero motion. Skipping frames here
    // previously jumped past V1_01_easy's takeoff, forcing a mid-flight
    // gravity estimate.
    static constexpr uint32_t kInitialFramesToSkip = 0;

    AnyBag& storage_;
    std::generator<ReadingType> generator_;
    std::optional<decltype(generator_.begin())> iter_;
    data::ReadingFilter filter_;

    // Items of each kind seen so far, kept or not
    uint64_t frame_count_ = 0;
    uint64_t imu_count_ = 0;

    // Set once the stream is exhausted or the filter ended it
    bool stopped_ = false;

    [[nodiscard]] std::optional<std::string> skipInitialFrames(uint32_t count);

    // Advances to the next frame the filter keeps, collecting the kept IMU
    // samples on the way. The error is a message, or kComputeStopExecution
    // when the stream ended
    [[nodiscard]] std::expected<data::Reading<1>, std::string> findNextFrame(
        std::vector<data::IMUReading>& imu);

    // Loads the reading's frames and publishes them with their timestamp and
    // the IMU samples
    [[nodiscard]] std::optional<std::string> publish(
        data::Reading<1> reading, std::vector<data::IMUReading> imu);

    // Asks the filter about the next item of `kind`. Keeps everything when
    // there is no filter
    [[nodiscard]] std::expected<data::ReadingDecision, std::string> filter(
        data::ReadingCandidate::Kind kind, uint64_t timestamp);

    // Marks the stream as ended and returns kComputeStopExecution
    [[nodiscard]] std::string stop(std::string_view reason);
};
};  // namespace wslam
