#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/gpu.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {

// When the Previous pyramid (the model's reference image) moves on to the
// frame held in Current
enum class ReferenceAdvance : uint8_t {
    // Every frame, so the model pairs consecutive frames. For runs without a
    // mapping stage
    EveryFrame,
    // Only after a frame for which the mapping stage published
    // ResourceIdentifier::FeatureReferenceAdvanceName (consumed here), so the
    // model pairs the last keyframe with the current frame
    OnKeyframeMarker,
};

// Copies every level of the Current pyramid into the Previous one, so later
// frames are matched against it. Runs at the start of a frame, before Current
// is refilled: whether frame N becomes the reference is decided by the mapping
// stage, after the feature detect stage of frame N has run
class PreservePyramidPass : public compute::GPUPass {
   public:
    PreservePyramidPass(std::shared_ptr<compute::GPU> gpu,
                        std::shared_ptr<SharedBindings> bindings,
                        AnyBag& storage, ReferenceAdvance advance)
        : GPUPass(std::move(gpu)),
          bindings_(std::move(bindings)),
          storage_(storage),
          advance_(advance) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> submitWork() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;
    AnyBag& storage_;
    const ReferenceAdvance advance_;

    // Whether Current holds a frame, false until the first frame was filled.
    // The fill pass follows this one in the same stage
    bool has_current_ = false;

    [[nodiscard]] bool shouldAdvance();
};

}  // namespace wslam::v2
