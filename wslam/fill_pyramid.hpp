#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <span>

#include "awaiter.hpp"
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
          image_getter_(std::move(opts.image_getter)),
          shared_bindings_(shared_bindings) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    ImageProvider image_getter_;

    GpuSharedBindings& shared_bindings_;
    wgpu::Sampler sampler_;

    // Two per texture view pair, one per bind group, one per compute pass
    static constexpr auto kTotalLabels = GPUConst::levels_of_detail * 3;

    // Has initial size of kTotalLabels
    std::vector<std::string> labels_ = std::invoke([]() {
        std::vector<std::string> v;
        v.reserve(kTotalLabels);
        return v;
    });

    std::array<std::pair<wgpu::TextureView, wgpu::TextureView>,
               GPUConst::levels_of_detail>
        texture_views_;
    std::array<wgpu::BindGroup, GPUConst::levels_of_detail - 1> bind_groups_;

    [[nodiscard]] std::optional<std::string> initBindGroupLayout();
    [[nodiscard]] std::optional<std::string> initSampler();
    [[nodiscard]] std::optional<std::string> initBindGroups();
    [[nodiscard]] std::optional<std::string> initTextureViews();
    [[nodiscard]] std::optional<std::string> initComputePipeline();

    [[nodiscard]] std::optional<std::string> writeBaseLayer();
    [[nodiscard]] std::optional<std::string> writeNonBaseLayers();
    void writeLayerN(compute::Awaiter& awaiter, wgpu::CommandEncoder& encoder,
                     size_t lod);
};

};  // namespace wslam
