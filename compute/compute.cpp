#include "compute.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <print>
#include <string_view>

#include "stage.hpp"
using namespace wslam::compute;

void impl::print_device_unresponsive(const wgpu::Device& device,
                                     wgpu::DeviceLostReason reason,
                                     wgpu::StringView message) {
    (void)device;
    if (reason == wgpu::DeviceLostReason::Destroyed) {
        // Deliberate teardown at shutdown — not an error.
        spdlog::debug("WebGPU device destroyed: {}", std::string_view(message));
        return;
    }
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

void Compute::addStage(Stage stage) {
    stage.storage_ = &storage_;
    stages_.emplace_back(std::move(stage));
}

PreinitOpts wslam::compute::createPreInitializeOpts() {
    const char* shader_dir = std::getenv(WSLAM_SHADER_SRC_DIR_ENV);
    if (shader_dir == nullptr) {
        shader_dir = "";
    }

    return {
        .deviceLostCallback_ = impl::print_device_unresponsive,
        .errorCallback_ = impl::print_device_captured_error,
        .shader_module_path_prefix_ = shader_dir,
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

    return std::nullopt;
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
            if (err.value() == kComputeStopExecution) {
                spdlog::info("[COMPUTE] Execution stop requested from stage {}",
                             stage.getId());

                return {};
            }

            return std::format("executing stage {}: {}", stage.getId(),
                               err.value());
        }
    }

    return std::nullopt;
}
