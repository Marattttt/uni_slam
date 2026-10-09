#pragma once

#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {

// TEMPORARY: reads the model's output keypoints back to the CPU, logs a
// summary and publishes them to storage. Only runs on frames the model ran
// on
class LoadKeypointsPass : public compute::Pass {
   public:
    LoadKeypointsPass(std::shared_ptr<compute::GPU> gpu,
                      std::shared_ptr<SharedBindings> bindings, AnyBag& storage)
        : gpu_(std::move(gpu)),
          bindings_(std::move(bindings)),
          storage_(storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<compute::GPU> gpu_;
    std::shared_ptr<SharedBindings> bindings_;
    AnyBag& storage_;
};

}  // namespace wslam::v2
