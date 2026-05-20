#pragma once

#include <webgpu/webgpu_cpp.h>

#include <memory>
#include <variant>

#include "anybag.hpp"
#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
namespace compute {

class Compute;

constexpr std::string kStageStopExecution = "STAGE_STOP";

class Stage {
    friend class Compute;

   public:
    using PassPtr
        = std::variant<std::unique_ptr<Pass>, std::unique_ptr<GPUPass>>;

    Stage(std::string id, std::shared_ptr<GPU> gpu);

    [[nodiscard]] std::string getId() const;
    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] std::optional<std::string> execute();

    void add_pass(PassPtr pass);
    void add_pass(std::vector<std::unique_ptr<Pass>> pass);

    AnyBag* storage_;

   private:
    std::vector<PassPtr> passes_;
    std::shared_ptr<GPU> gpu_;
    std::string id_;

    [[nodiscard]] std::optional<std::string> executeGPUBatch(
        std::span<GPUPass*> batch);
};
};  // namespace compute
};  // namespace wslam
