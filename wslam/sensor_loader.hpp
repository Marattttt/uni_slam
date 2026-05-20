#pragma once

#include "anybag.hpp"
#include "pass.hpp"
#include "provider_base.hpp"

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
    // Number of initial frames to drop on the first execute(). These come from
    // a cold sensor/dataset and are often stale or low-quality, so skipping
    // them stabilises downstream feature matching.
    static constexpr uint32_t kInitialFramesToSkip = 200;

    AnyBag& storage_;
    std::generator<ReadingType> generator_;
    std::optional<decltype(generator_.begin())> iter_;

    [[nodiscard]] std::optional<std::string> skipInitialFrames(uint32_t count);
};
};  // namespace wslam
