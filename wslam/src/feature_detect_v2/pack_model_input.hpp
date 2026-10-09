#pragma once

#include <webgpu/webgpu_cpp.h>

#include <memory>
#include <optional>
#include <string>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {

// Packs the Previous and Current pyramids' model level into the model's
// images buffer (raco_lightglue::kImagesShape). Records nothing until a
// Previous pyramid exists
class PackModelInputPass : public compute::GPUPass {
   public:
    PackModelInputPass(std::shared_ptr<compute::GPU> gpu,
                       std::shared_ptr<SharedBindings> bindings)
        : GPUPass(std::move(gpu)), bindings_(std::move(bindings)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> prepareExecute(
        const wgpu::CommandEncoder& encoder) override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;

    wgpu::TextureView left_view_;
    wgpu::TextureView right_view_;
    wgpu::BindGroupLayout bind_group_layout_;
    wgpu::BindGroup bind_group_;
    wgpu::ComputePipeline compute_pipeline_;

    [[nodiscard]] std::optional<std::string> initTextureViews();
    [[nodiscard]] std::optional<std::string> initBindGroupLayout();
    [[nodiscard]] std::optional<std::string> initBindGroup();
    [[nodiscard]] std::optional<std::string> initComputePipeline();
};

}  // namespace wslam::v2
