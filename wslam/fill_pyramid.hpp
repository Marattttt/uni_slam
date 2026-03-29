#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <span>

#include "common.hpp"
#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
using ImageProvider
    = std::function<std::optional<std::span<const std::byte>>()>;

struct PassFillPyramidOpts {
    ImageProvider image_getter;
};

class FillPyramidPass : public compute::GPUPass {
   public:
    FillPyramidPass(std::shared_ptr<compute::GPU> gpu,
                    GpuSharedBindings& shared_bindings,
                    PassFillPyramidOpts opts)
        : GPUPass(std::move(gpu)),
          shared_bindings_(shared_bindings),
          image_getter_(std::move(opts.image_getter)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    GpuSharedBindings& shared_bindings_;
    ImageProvider image_getter_;
    wgpu::Sampler sampler_;

    [[nodiscard]] std::optional<std::string> initBindGroupLayout();
    [[nodiscard]] std::optional<std::string> initSampler();
    [[nodiscard]] std::optional<std::string> initBindGroup();
    [[nodiscard]] std::optional<std::string> initComputePipeline();

    [[nodiscard]] std::optional<std::string> writeBaseMip();
    [[nodiscard]] std::optional<std::string> writePyramidMips();
};

};  // namespace wslam
