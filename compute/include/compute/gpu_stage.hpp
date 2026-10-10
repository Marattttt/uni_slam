#pragma once

#include <webgpu/webgpu_cpp.h>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "compute/performance.hpp"
#include "compute/stage.hpp"

namespace wslam {
namespace compute {

// Stage runs consecutive GPU passes back to back, each submitting its own work,
// and waits once for the whole run. Still supports regular CPU passes, waiting
// for the GPU passes before each of them
class GpuStage : public Stage {
   public:
    using PassPtr
        = std::variant<std::unique_ptr<Pass>, std::unique_ptr<GPUPass>>;

    GpuStage(std::string id, std::shared_ptr<GPU> gpu,
             PerfRecorder* perf = nullptr);

    // CPU passes are stored in the same ordered list as GPU passes so batching
    // respects insertion order; hides the base add_pass by design.
    void add_pass(std::unique_ptr<Pass> pass);
    void add_pass(std::unique_ptr<GPUPass> pass);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

   private:
    // Walks the pass list, batching consecutive GPU passes and flushing before
    // each CPU pass; honours the base sentinel outcomes.
    [[nodiscard]] std::optional<std::string> runBatched();
    [[nodiscard]] std::optional<std::string> executeGPUBatch(
        std::span<GPUPass*> batch);

    std::vector<PassPtr> passes_;
    std::shared_ptr<GPU> gpu_;
};
};  // namespace compute
};  // namespace wslam
