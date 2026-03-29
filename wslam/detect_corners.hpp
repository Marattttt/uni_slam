#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>

#include "common.hpp"
#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
class PassDetectCorners : public compute::GPUPass {
   public:
    PassDetectCorners(std::shared_ptr<compute::GPU> gpu,
                      GpuSharedBindings& shared_bindings)
        : GPUPass(std::move(gpu)), shared_bindings_(shared_bindings) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    const uint16_t kFrameW = GPUConst::frame_width;
    const uint16_t kFrameH = GPUConst::frame_height;
    const uint8_t kPyramidLevels = GPUConst::pyr_levels;
    const uint64_t kImageBufSize
        = static_cast<uint64_t>(kFrameW) * kFrameH * sizeof(uint8_t);

    GpuSharedBindings& shared_bindings_;
    compute::GPU::BufferBindingMap buf_bindings_;

    std::optional<std::string> initBindGroup();
    std::optional<std::string> initBindGroupLayout();
    std::optional<std::string> initComputePipeline();
};
};  // namespace wslam
