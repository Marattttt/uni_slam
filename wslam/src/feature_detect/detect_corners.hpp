#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>

#include "compute/awaiter.hpp"
#include "common.hpp"
#include "compute/gpu.hpp"
#include "models.hpp"
#include "compute/pass.hpp"

namespace wslam {
class PassDetectCorners : public compute::GPUPass {
   public:
    PassDetectCorners(std::shared_ptr<compute::GPU> gpu,
                      GpuSharedBindings& shared_bindings,
                      std::string output_label)
        : GPUPass(std::move(gpu)),
          kCornersOutputLabel(std::move(output_label)),
          shared_bindings_(shared_bindings) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> prepareExecute(
        const wgpu::CommandEncoder& encoder) override;

   private:
    struct PassParams {
        uint32_t lod;
    };

    static constexpr size_t kConrnersArraySize
        = sizeof(std::array<gpumodels::CornersBlock<GPUConst::frame_width,
                                                    GPUConst::frame_height>,
                            GPUConst::levels_of_detail>);

    const std::string kCornersOutputLabel;

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

    void fillPassParams();

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

    void saveOutputBindings();
};
};  // namespace wslam
