#pragma once

#include <webgpu/webgpu_cpp.h>

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>

#include "gpu.hpp"

namespace wslam::compute {

using BindingMap
    = std::unordered_map<std::string,
                         std::variant<BufferBinding, wgpu::Texture>>;

class Pass {
   public:
    Pass(std::shared_ptr<GPU> gpu) : gpu_(std::move(gpu)) {}
    virtual ~Pass() = default;

    [[nodiscard]] virtual std::optional<std::string> initialize() = 0;
    [[nodiscard]] virtual std::optional<std::string> execute() = 0;
    [[nodiscard]] virtual std::string getId() const = 0;

   protected:
    std::shared_ptr<GPU> gpu_;
};

class GPUPass : public Pass {
   public:
    GPUPass(std::shared_ptr<GPU> gpu) : Pass(std::move(gpu)) {}
    [[nodiscard]] virtual BindingMap getOutputBindings();

   protected:
    wgpu::BindGroup bind_group_;
    wgpu::BindGroupLayout bind_group_layout_;
    wgpu::ComputePipeline compute_pipeline_;
    wgpu::PipelineLayout compute_pipeline_layout_;
};

class CustomPass : public Pass {
   public:
    using Callback = std::function<std::optional<std::string>(CustomPass*)>;

    CustomPass(std::shared_ptr<GPU> gpu, Callback callback)
        : Pass(std::move(gpu)), callback_(std::move(callback)) {}

    std::optional<std::string> initialize() override;

    std::optional<std::string> execute() override;

   protected:
    Callback callback_;
};
}  // namespace wslam::compute
;  // namespace wslam
