#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

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

enum class DefinedWorkflow : uint8_t { None, HelloWGSL };

struct PreinitOpts {
    const std::function<wgpu::DeviceLostCallback<void>> deviceLostCallback_;
    const std::function<wgpu::UncapturedErrorCallback<void>> errorCallback_;
    const std::filesystem::path shader_module_path_prefix_;
    DefinedWorkflow workflow;
};

PreinitOpts createPreInitializeOpts(DefinedWorkflow = DefinedWorkflow::None);

class Compute {
   public:
    void print_adapter_info() const;

    [[nodiscard]] std::optional<std::string> preInitialize(
        const PreinitOpts& opts = createPreInitializeOpts());
    [[nodiscard]] std::optional<std::string> initizalizeAllStages();
    [[nodiscard]] std::optional<std::string> execute();

    void addStage(Stage stage);
    GPU& getGPU();
    std::shared_ptr<GPU> getGPUPtr();

   private:
    void createStages(DefinedWorkflow workflow);

    std::shared_ptr<GPU> gpu_;
    std::vector<Stage> stages_;
};
};  // namespace compute
};  // namespace wslam
