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

// Which of the two image pyramids
enum class PyramidSlot : uint8_t { Current, Previous };

// GPU resources shared by the passes of the v2 feature detect stage. Owned
// jointly by those passes through a shared_ptr. keypoint_count is the model's
// keypoints per image, fixed at startup
class SharedBindings {
   public:
    SharedBindings(std::shared_ptr<compute::GPU> gpu, int64_t keypoint_count)
        : gpu_(std::move(gpu)), keypoint_count_(keypoint_count) {
        WSLAM_ASSERT(keypoint_count > 0, "keypoint count is {}",
                     keypoint_count);
    }

    [[nodiscard]] std::optional<std::string> initialize();

    [[nodiscard]] const wgpu::Texture& getTexture(PyramidSlot slot,
                                                  size_t lod) const {
        return pyramids_.at(static_cast<size_t>(slot)).at(lod);
    }

    // Model input, raco_lightglue::kImagesShape
    [[nodiscard]] const wgpu::Buffer& getImagesBuffer() const {
        return images_buffer_;
    }
    // Model output, raco_lightglue::KeypointsShape(getKeypointCount())
    [[nodiscard]] const wgpu::Buffer& getKeypointsBuffer() const {
        return keypoints_buffer_;
    }

    [[nodiscard]] int64_t getKeypointCount() const { return keypoint_count_; }

    // Whether the Previous pyramid holds a frame. False until the first frame
    // has been preserved
    [[nodiscard]] bool hasPrevious() const { return has_previous_; }
    void markPreviousValid() { has_previous_ = true; }

   private:
    std::shared_ptr<compute::GPU> gpu_;
    const int64_t keypoint_count_;

    std::array<std::array<wgpu::Texture, GPUConst::levels_of_detail>, 2>
        pyramids_;
    wgpu::Buffer images_buffer_;
    wgpu::Buffer keypoints_buffer_;

    bool has_previous_ = false;

    [[nodiscard]] std::optional<std::string> initTextures();
    [[nodiscard]] std::optional<std::string> initBuffers();
    void initTexture(compute::Awaiter& awaiter, PyramidSlot slot, uint32_t lod);
};

}  // namespace wslam::v2
