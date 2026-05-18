#pragma once

#include "common.hpp"
#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
class MatchFeaturesCPU : public compute::Pass {
   public:
    MatchFeaturesCPU(GpuSharedBindings& shared,
                     std::shared_ptr<compute::GPU> gpu)
        : compute::Pass(std::move(gpu)), shared_(shared) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    GpuSharedBindings& shared_;
};
}  // namespace wslam
