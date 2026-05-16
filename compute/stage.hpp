#pragma once

#include <webgpu/webgpu_cpp.h>

#include <memory>

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
    Stage(std::string id, std::shared_ptr<GPU> gpu);
    std::optional<std::string> initialize();
    std::optional<std::string> execute();

    void add_pass(std::unique_ptr<Pass> pass);
    void add_pass(std::vector<std::unique_ptr<Pass>> pass);

    [[nodiscard]] std::string getId() const;

    AnyBag* storage_;

   private:
    std::vector<std::unique_ptr<Pass>> passes_;
    std::shared_ptr<GPU> gpu_;
    std::string id_;
};

Stage CreateHelloWgslStage(const std::shared_ptr<GPU>& gpu);
};  // namespace compute
};  // namespace wslam
