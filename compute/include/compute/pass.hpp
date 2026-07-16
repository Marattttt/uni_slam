#pragma once

#include <webgpu/webgpu_cpp.h>

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>

#include "compute/gpu.hpp"

namespace wslam::compute {

using BindingMap
    = std::unordered_map<std::string,
                         std::variant<BufferBinding, wgpu::Texture>>;

class Pass {
   public:
    Pass() = default;
    virtual ~Pass() = default;

    [[nodiscard]] virtual std::string getId() const = 0;
    [[nodiscard]] virtual std::optional<std::string> initialize() = 0;
    [[nodiscard]] virtual std::optional<std::string> execute() = 0;
};

class GPUPass {
   public:
    GPUPass(std::shared_ptr<GPU> gpu) : gpu_(std::move(gpu)) {}
    virtual ~GPUPass() = default;

    [[nodiscard]] virtual std::string getId() const = 0;
    [[nodiscard]] virtual std::optional<std::string> initialize() = 0;
    [[nodiscard]] virtual std::optional<std::string> prepareExecute(
        const wgpu::CommandEncoder& encoder) = 0;

   protected:
    std::shared_ptr<GPU> gpu_;
};

class CustomPass : public Pass {
   public:
    using Callback = std::function<std::optional<std::string>(CustomPass*)>;

    CustomPass(std::string id, Callback callback)
        : callback_(std::move(callback)), id_(std::move(id)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    Callback callback_;
    const std::string id_;
};
}  // namespace wslam::compute
;  // namespace wslam
