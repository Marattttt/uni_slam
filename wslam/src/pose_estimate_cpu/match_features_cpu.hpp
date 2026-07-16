#pragma once

#include "common.hpp"
#include "compute/pass.hpp"

namespace wslam {
class MatchFeaturesCPU : public compute::Pass {
   public:
    MatchFeaturesCPU(AnyBag& storage) : storage_(storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
};
}  // namespace wslam
