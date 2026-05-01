#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <unordered_map>
#include <utility>

#include "common.hpp"
#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
class CullCornersPass : public compute::Pass {
   public:
    CullCornersPass(std::shared_ptr<compute::GPU> gpu,
                    GpuSharedBindings& shared_bindings,
                    std::string output_label, std::string corners_label)

        : Pass(std::move(gpu)),
          kCulledCornersOutputLabel(std::move(output_label)),
          kInputCornersLabel(std::move(corners_label)),
          shared_bindings_(shared_bindings) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    enum class Workflow : uint8_t { Horizontal, Vertical };

    static constexpr size_t kPassCount = 2;
    static constexpr std::string_view kVerticalBindingLabel = "vertical";
    static constexpr std::string_view kShaderModulePath = "cull_corners.wgsl";

    const std::string kCulledCornersOutputLabel;
    const std::string kInputCornersLabel;

    GpuSharedBindings& shared_bindings_;
    compute::GPU::BufferBindingMap buf_bindings_;

    struct PassGpuData {
        wgpu::BindGroup bg;
        wgpu::BindGroupLayout bg_layout;
        wgpu::ComputePipeline pipeline;
        wgpu::PipelineLayout pipeline_layout;
    };

    std::unordered_map<Workflow, PassGpuData> gpu_data_;
    wgpu::BindGroupLayout bind_group_layout_;
    wgpu::PipelineLayout compute_pipeline_layout_;
    std::array<wgpu::ConstantEntry, 2> compute_constants_;

    void initGpuDataEntries();
    void initComputeConstants();
    [[nodiscard]] std::optional<std::string> initBindingLayout();
    [[nodiscard]] std::optional<std::string> initBindGroups();
    [[nodiscard]] std::optional<std::string> initComputePipeline();
};
};  // namespace wslam
