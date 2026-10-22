#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "assert.hpp"
#include "compute/gpu.hpp"

namespace wslam::v2 {

// GPU resources shared by the passes of the v2 feature detect stage: the
// current frame's image and the model's input. Owned jointly by those passes
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

    // The current frame, at camera resolution
    [[nodiscard]] const wgpu::Texture& getSourceTexture() const {
        return source_texture_;
    }

    // Model input, raco_lightglue::kImagesShape
    [[nodiscard]] const wgpu::Buffer& getImagesBuffer() const {
        return images_buffer_;
    }

    [[nodiscard]] int64_t getKeypointCount() const { return keypoint_count_; }

   private:
    std::shared_ptr<compute::GPU> gpu_;
    const int64_t keypoint_count_;

    wgpu::Texture source_texture_;
    wgpu::Buffer images_buffer_;

    [[nodiscard]] std::optional<std::string> initTexture();
    [[nodiscard]] std::optional<std::string> initBuffer();
};

}  // namespace wslam::v2
