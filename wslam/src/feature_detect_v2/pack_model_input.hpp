#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {
using ImageProvider
    = std::function<std::optional<std::span<const std::byte>>()>;

// Uploads the current frame into the shared source texture and packs it into
// the model's images buffer (raco_lightglue::kImagesShape). The model runs at
// camera resolution, so there is no image pyramid
class PackModelInputPass : public compute::GPUPass {
   public:
    PackModelInputPass(std::shared_ptr<compute::GPU> gpu,
                       std::shared_ptr<SharedBindings> bindings,
                       ImageProvider image_getter)
        : GPUPass(std::move(gpu)),
          bindings_(std::move(bindings)),
          image_getter_(std::move(image_getter)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> submitWork() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;
    ImageProvider image_getter_;

    wgpu::TextureView view_;
    wgpu::BindGroupLayout bind_group_layout_;
    wgpu::BindGroup bind_group_;
    wgpu::ComputePipeline compute_pipeline_;

    [[nodiscard]] std::optional<std::string> initTextureViews();
    [[nodiscard]] std::optional<std::string> initBindGroupLayout();
    [[nodiscard]] std::optional<std::string> initBindGroup();
    [[nodiscard]] std::optional<std::string> initComputePipeline();

    [[nodiscard]] std::optional<std::string> writeSource();
};

}  // namespace wslam::v2
