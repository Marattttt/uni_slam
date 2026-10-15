#pragma once

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <memory>
#include <optional>
#include <string>

#include "assert.hpp"
#include "common.hpp"
#include "compute/awaiter.hpp"
#include "compute/gpu.hpp"

namespace wslam::v2 {

// GPU resources shared by the passes of the v2 feature detect stage: the
// current frame's pyramid and the model's input. Owned jointly by those passes
// through a shared_ptr. keypoint_count is the model's keypoints per image,
// fixed at startup
class SharedBindings {
   public:
    SharedBindings(std::shared_ptr<compute::GPU> gpu, int64_t keypoint_count)
        : gpu_(std::move(gpu)), keypoint_count_(keypoint_count) {
        WSLAM_ASSERT(keypoint_count > 0, "keypoint count is {}",
                     keypoint_count);
    }

    [[nodiscard]] std::optional<std::string> initialize();

    // Level lod of the current frame's pyramid
    [[nodiscard]] const wgpu::Texture& getTexture(size_t lod) const {
        return pyramid_.at(lod);
    }

    // Model input, raco_lightglue::kImagesShape
    [[nodiscard]] const wgpu::Buffer& getImagesBuffer() const {
        return images_buffer_;
    }

    [[nodiscard]] int64_t getKeypointCount() const { return keypoint_count_; }

   private:
    std::shared_ptr<compute::GPU> gpu_;
    const int64_t keypoint_count_;

    std::array<wgpu::Texture, GPUConst::levels_of_detail> pyramid_;
    wgpu::Buffer images_buffer_;

    [[nodiscard]] std::optional<std::string> initTextures();
    [[nodiscard]] std::optional<std::string> initBuffers();
    void initTexture(compute::Awaiter& awaiter, uint32_t lod);
};

}  // namespace wslam::v2
