#pragma once

#include "gpu.hpp"
#include "pass.hpp"

namespace wslam::compute {
class HelloWGSLPass : public GPUPass {
   public:
    HelloWGSLPass(std::shared_ptr<GPU> gpu) : GPUPass(std::move(gpu)) {}
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    static const uint64_t kBufSize = 256;
    GPU::BufferBindingMap buf_bindings_;

    std::optional<std::string> initBindGroup();
    std::optional<std::string> initBindGroupLayout();
    std::optional<std::string> initComputePipeline();
};
};  // namespace wslam::compute
