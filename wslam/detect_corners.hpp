#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>

#include "awaiter.hpp"
#include "common.hpp"
#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
class PassDetectCorners : public compute::Pass {
   public:
    struct PassParams {
        uint32_t lod;
    };

    PassDetectCorners(std::shared_ptr<compute::GPU> gpu,
                      GpuSharedBindings& shared_bindings)
        : Pass(std::move(gpu)), shared_bindings_(shared_bindings) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    const size_t kConrnersArraySize
        = static_cast<size_t>(GPUConst::frame_width) * GPUConst::frame_height
          * GPUConst::pixel_size * GPUConst::levels_of_detail;

    GpuSharedBindings& shared_bindings_;
    compute::GPU::BufferBindingMap buf_bindings_;

    std::array<wgpu::TextureView, GPUConst::levels_of_detail> texture_views_;

    struct BindGroupLayouts {
        wgpu::BindGroupLayout corners;
        wgpu::BindGroupLayout pass_data;
        // Methods to allow use as an array of bindgroups (makes things prettier
        // I think)
        static constexpr size_t size() { return 2; }
        wgpu::BindGroupLayout* data() { return &corners; }
    };
    BindGroupLayouts bind_group_layouts_;
    std::array<wgpu::BindGroup, GPUConst::levels_of_detail>
        per_pass_bind_groups_;
    wgpu::BindGroup common_bind_group_;
    std::array<PassParams, GPUConst::levels_of_detail> pass_params_;

    wgpu::PipelineLayout compute_pipeline_layout_;
    wgpu::ComputePipeline compute_pipeline_;
    std::array<wgpu::ConstantEntry, 3> compute_constants;

    void fillPassParams();
    void fillConstants();

    [[nodiscard]] std::optional<std::string> initTextureViews();
    [[nodiscard]] std::optional<std::string> initPerPassBindGroups();
    [[nodiscard]] std::optional<std::string> initCommonBindGroup();
    [[nodiscard]] std::optional<std::string> initComputePipeline();

    [[nodiscard]] std::optional<std::string> initBindGroupLayouts();
    [[nodiscard]] std::optional<std::string> initCornersBindGroupLayout(
        compute::Awaiter&);
    [[nodiscard]] std::optional<std::string> initPassDataBindGroupLayout(
        compute::Awaiter&);

    [[nodiscard]] std::optional<std::string> writeGPUPassParams(
        const wgpu::CommandEncoder&);
};
};  // namespace wslam
