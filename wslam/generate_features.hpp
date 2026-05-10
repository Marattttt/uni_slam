#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <expected>
#include <memory>

#include "common.hpp"
#include "gpu.hpp"
#include "models.hpp"
#include "pass.hpp"

namespace wslam {
class GenerateFeaturesPass : public compute::Pass {
   public:
    GenerateFeaturesPass(std::shared_ptr<compute::GPU> gpu,
                         GpuSharedBindings& shared_bindings,
                         std::string corners_label, std::string output_label)
        : Pass(std::move(gpu)),
          kCornersLabel(std::move(corners_label)),
          kFeaturesLabel(std::move(output_label)),
          shared_(shared_bindings) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    const std::string kCornersLabel;
    const std::string kFeaturesLabel;

    static constexpr auto kPassCount = GPUConst::levels_of_detail;

    static constexpr auto kCornersBindingSize
        = sizeof(gpumodels::CornersBlock<GPUConst::frame_width,
                                         GPUConst::frame_height>)
          * GPUConst::levels_of_detail;
    static constexpr auto kBriefTestsBindingSize
        = sizeof(std::array<std::array<int32_t, 4>, 256>);
    static constexpr auto kFeatureArrayBindingSize
        = sizeof(gpumodels::FeatureArray<>);

    GpuSharedBindings& shared_;
    std::optional<compute::BufferBinding> brief_tests_binding_;

    std::array<wgpu::BindGroupLayout, 2> bind_group_layouts_;
    std::array<std::array<wgpu::BindGroup, 2>, kPassCount> bind_groups_;
    wgpu::PipelineLayout compute_pipeline_layout_;
    wgpu::ComputePipeline compute_pipeline_;
    wgpu::Sampler sampler_;

    [[nodiscard]] std::expected<void, std::string> initSampler();
    [[nodiscard]] std::expected<void, std::string> initBindGroupLayouts();
    [[nodiscard]] std::expected<void, std::string> initBindGroups();
    [[nodiscard]] std::expected<void, std::string> initComputePipeline();

    [[nodiscard]] std::expected<void, std::string> initCommonBindgroupLayout();
    [[nodiscard]] std::expected<void, std::string> initPerPassBindgroupLayout();

    [[nodiscard]] std::expected<void, std::string> initCommonBindgroup();
    [[nodiscard]] std::expected<void, std::string> initPerPassBindgroups();

    [[nodiscard]] std::expected<void, std::string> writeBRIEFvalues();
};
}  // namespace wslam
