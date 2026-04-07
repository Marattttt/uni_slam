#pragma once

#include <memory>

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

    SensorLoaderPass(std::generator<ReadingType>&& generator,
                     std::shared_ptr<compute::GPU> gpu, AnyBag& storage)
        : compute::Pass(std::move(gpu)),
          storage_(storage),
          generator_(std::move(generator)) {};

   private:
    AnyBag& storage_;
    std::generator<ReadingType> generator_;
    std::optional<decltype(generator_.begin())> iter_;
};
};  // namespace wslam
