#include "compute.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <print>
#include <string_view>

#include "pass_hellowgsl.hpp"
#include "stage.hpp"
using namespace wslam::compute;

void impl::print_device_unresponsive(const wgpu::Device& device,
                                     wgpu::DeviceLostReason reason,
                                     wgpu::StringView message) {
    (void)device;
    std::println(stderr, "Lost connection with device. Reason: {}; Message: {}",
                 static_cast<uint32_t>(reason), std::string_view(message));
}

void impl::print_device_captured_error(const wgpu::Device& device,
                                       wgpu::ErrorType errtype,
                                       wgpu::StringView msg) {
    (void)device;
    std::println(
        stderr,
        "Uncaptured error for WebGPU device. ErrorType: {}; Message: {}",
        static_cast<uint32_t>(errtype), std::string_view(msg));
}

PreinitOpts wslam::compute::createPreInitializeOpts(DefinedWorkflow workflow) {
    const char* shader_dir = std::getenv(WSLAM_SHADER_SRC_DIR_ENV);
    if (shader_dir == nullptr) {
        shader_dir = "";
    }

    return {
        .deviceLostCallback_ = impl::print_device_unresponsive,
        .errorCallback_ = impl::print_device_captured_error,
        .shader_module_path_prefix_ = shader_dir,
        .workflow = workflow,
    };
}

std::optional<std::string> Compute::preInitialize(const PreinitOpts& opts) {
    gpu_ = std::make_shared<GPU>(
        std::make_shared<GPU::DeviceCb>(opts.deviceLostCallback_),
        std::make_shared<GPU::ErrorCb>(opts.errorCallback_),
        opts.shader_module_path_prefix_);

    std::optional<std::string> err;

    err = gpu_->initialize();
    if (err.has_value()) {
        return std::format("resources: {}", err.value());
    }

    spdlog::debug("[Compute] initializing for hello wslam");
    createStages(DefinedWorkflow::HelloWGSL);

    return std::nullopt;
}

void Compute::createStages(DefinedWorkflow workflow) {
    std::unique_ptr<Stage> stage;
    switch (workflow) {
        case DefinedWorkflow::HelloWGSL:
            stage = std::make_unique<Stage>("Hello wgsl", gpu_);
            stage->add_pass(std::make_unique<HelloWGSLPass>(gpu_));

            stages_.emplace_back(std::move(*stage));
            break;
        case DefinedWorkflow::None:
            break;
        default:
            throw std::logic_error(
                std::format("invalid workflow {}", static_cast<int>(workflow)));
    }
}

std::optional<std::string> Compute::initizalizeAllStages() {
    for (Stage& stage : stages_) {
        if (auto err = stage.initialize()) {
            return std::format("iniiializing stage {}: {}", stage.getId(),
                               err.value());
        }
    }

    return std::nullopt;
}

std::optional<std::string> Compute::execute() {
    for (Stage& stage : stages_) {
        if (auto err = stage.execute()) {
            return std::format("executing stage {}: {}", stage.getId(),
                               err.value());
        }
    }

    return std::nullopt;
}

void Compute::addStage(Stage stage) { stages_.emplace_back(std::move(stage)); }
GPU& Compute::getGPU() { return *gpu_; }
std::shared_ptr<GPU> Compute::getGPUPtr() { return gpu_; };
