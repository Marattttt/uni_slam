#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {
using ImageProvider
    = std::function<std::optional<std::span<const std::byte>>()>;

struct PassSourceImageOpts {
    ImageProvider image_getter;
};

// Writes the current frame into the shared source texture, the image the model
// input packer crops and packs. The model runs at camera resolution, so there
// is no image pyramid
class SourceImagePass : public compute::GPUPass {
   public:
    SourceImagePass(std::shared_ptr<compute::GPU> gpu,
                    std::shared_ptr<SharedBindings> bindings,
                    PassSourceImageOpts opts)
        : GPUPass(std::move(gpu)),
          image_getter_(std::move(opts.image_getter)),
          bindings_(std::move(bindings)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> submitWork() override;
    [[nodiscard]] std::string getId() const override;

   private:
    ImageProvider image_getter_;
    std::shared_ptr<SharedBindings> bindings_;

    [[nodiscard]] std::optional<std::string> writeSource();
};

}  // namespace wslam::v2
