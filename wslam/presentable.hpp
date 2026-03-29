#pragma once

#include "common.hpp"
#include "detect_corners.hpp"
#include "fill_pyramid.hpp"
#include "stage.hpp"

namespace wslam {
class Presentation : public compute::Stage {
   public:
    Presentation(const std::shared_ptr<compute::GPU>& gpu,
                 ImageProvider image_provider)
        : compute::Stage("Presentation", gpu),
          gpu_(gpu),
          fill_pyramid_(gpu, shared_,
                        {.image_getter = std::move(image_provider)}),
          detect_corners_(gpu, shared_) {}

    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] std::optional<std::string> execute();

   private:
    GpuSharedBindings shared_;
    std::shared_ptr<compute::GPU> gpu_;

    FillPyramidPass fill_pyramid_;
    PassDetectCorners detect_corners_;
};

}  // namespace wslam
