#pragma once

#include <webgpu/webgpu_cpp.h>

#include <memory>

#include "gpu.hpp"
#include "pass.hpp"

namespace wslam {
namespace compute {
class Stage {
   public:
    Stage(std::string type, std::shared_ptr<GPU> gpu);
    std::optional<std::string> initialize();
    std::optional<std::string> execute();

    void add_pass(std::unique_ptr<Pass> pass);
    void add_pass(std::vector<std::unique_ptr<Pass>> pass);

    [[nodiscard]] std::string getId() const;

   private:
    std::vector<std::unique_ptr<Pass>> passes_;
    std::shared_ptr<GPU> gpu_;
    std::string stage_type_;
};

Stage CreateHelloWgslStage(const std::shared_ptr<GPU>& gpu);
};  // namespace compute
};  // namespace wslam
