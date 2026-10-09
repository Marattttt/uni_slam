#pragma once

#include <webgpu/webgpu_cpp.h>

#include <memory>
#include <optional>
#include <string>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {

// Copies every level of the Current pyramid into the Previous one, so the
// next frame is matched against this one
class PreservePyramidPass : public compute::GPUPass {
   public:
    PreservePyramidPass(std::shared_ptr<compute::GPU> gpu,
                        std::shared_ptr<SharedBindings> bindings)
        : GPUPass(std::move(gpu)), bindings_(std::move(bindings)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> prepareExecute(
        const wgpu::CommandEncoder& encoder) override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;
};

}  // namespace wslam::v2
