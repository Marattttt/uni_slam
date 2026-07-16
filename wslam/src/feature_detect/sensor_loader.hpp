#pragma once

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"

namespace wslam {
class SensorLoaderPass : public compute::Pass {
   public:
    using ReadingType = std::expected<data::Reading<1>, std::string>;

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    SensorLoaderPass(std::generator<ReadingType>&& generator, AnyBag& storage)
        : storage_(storage), generator_(std::move(generator)) {};

   private:
    // Number of initial frames to drop on the first execute(). Zero: the
    // mapping stage *needs* the dataset's stationary lead-in — gravity and
    // gyro-bias are initialised from a stationary IMU window and the first
    // keyframe's velocity prior assumes ~zero motion. Skipping frames here
    // previously jumped past V1_01_easy's takeoff, forcing a mid-flight
    // gravity estimate (benchmarks/ACCURACY_ANALYSIS.md R3).
    static constexpr uint32_t kInitialFramesToSkip = 0;

    AnyBag& storage_;
    std::generator<ReadingType> generator_;
    std::optional<decltype(generator_.begin())> iter_;

    [[nodiscard]] std::optional<std::string> skipInitialFrames(uint32_t count);
};
};  // namespace wslam
