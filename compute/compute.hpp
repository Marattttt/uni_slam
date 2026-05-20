#pragma once

#include <webgpu/webgpu_cpp.h>

#include <any>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "anybag.hpp"
#include "gpu.hpp"
#include "stage.hpp"

#ifndef WSLAM_SHADER_SRC_DIR_ENV
#define WSLAM_SHADER_SRC_DIR_ENV "WSLAM_SHADER_SRC_DIR"
#endif

namespace wslam {
namespace compute {
namespace impl {
void print_device_unresponsive(const wgpu::Device& device,
                               wgpu::DeviceLostReason reason,
                               wgpu::StringView message);
void print_device_captured_error(const wgpu::Device& device,
                                 wgpu::ErrorType errtype, wgpu::StringView msg);
};  // namespace impl

struct PreinitOpts {
    const std::function<wgpu::DeviceLostCallback<void>> deviceLostCallback_;
    const std::function<wgpu::UncapturedErrorCallback<void>> errorCallback_;
    const std::filesystem::path shader_module_path_prefix_;
};

PreinitOpts createPreInitializeOpts();

constexpr std::string kComputeStopExecution = "COMP_STOP";
constexpr std::string kFullStopExecution = "FULL_STOP";

class Compute {
   public:
    using StorageKeyType = std::string;
    using StorageItemType = std::any;

    void print_adapter_info() const;

    [[nodiscard]] std::optional<std::string> preInitialize(
        const PreinitOpts& opts = createPreInitializeOpts());
    [[nodiscard]] std::optional<std::string> initizalizeAllStages();
    [[nodiscard]] std::optional<std::string> execute();

    void addStage(Stage stage);

    constexpr AnyBag& getStorage() { return storage_; }
    constexpr const AnyBag& getStorage() const { return storage_; }
    constexpr GPU& getGPU() { return *gpu_; }
    constexpr GPU& getGPU() const { return *gpu_; }
    constexpr std::shared_ptr<GPU> getGPUPtr() { return gpu_; }
    constexpr std::shared_ptr<GPU> getGPUPtr() const { return gpu_; }

   private:
    std::shared_ptr<GPU> gpu_;
    std::vector<Stage> stages_;
    AnyBag storage_;
};
};  // namespace compute
};  // namespace wslam
