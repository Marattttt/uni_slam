#pragma once

#include <webgpu/webgpu_cpp.h>

#include <string>

#include "gpu.hpp"

#ifndef WSLAM_FRAME_WIDTH
#define WSLAM_FRAME_WIDTH 752  // Euroc mav
#endif

#ifndef WSLAM_FRAME_HEIGHT
#define WSLAM_FRAME_HEIGHT 480  // Euroc mav
#endif

namespace wslam {
class FillPyramidPass;

namespace GPUConst {
constexpr uint32_t frame_width = WSLAM_FRAME_WIDTH;
constexpr uint32_t frame_height = WSLAM_FRAME_HEIGHT;
constexpr uint32_t pixel_size = sizeof(float);
constexpr uint32_t levels_of_detail = 6;
constexpr double lod_scale_factor = 1.2;
};  // namespace GPUConst

namespace GPUBindingSize {
consteval size_t getPyramidLayerSize(const uint32_t lod) {
    if (lod >= GPUConst::levels_of_detail) {
        throw std::out_of_range("LoD out of range");
    }

    double scale = 1.0;

    for (size_t i = 0; i < lod; ++i) {
        scale *= GPUConst::lod_scale_factor;
    }

    const auto width = static_cast<size_t>(GPUConst::frame_width / scale);
    const auto height = static_cast<size_t>(GPUConst::frame_height / scale);

    return width * height * GPUConst::pixel_size;
}

// Size (int bytes) of the initial frame
constexpr size_t source_frame = static_cast<size_t>(GPUConst::frame_width)
                                * GPUConst::frame_height * GPUConst::pixel_size;

// Maximum space a LoD pyramid can take up
constexpr size_t max_pyramid_size = std::invoke([]() consteval {
    size_t total = 0;
    for (uint32_t i = 0; i < GPUConst::levels_of_detail; ++i) {
        total += getPyramidLayerSize(i);
    }
    return total;
});

}  // namespace GPUBindingSize

class GpuSharedBindings {
   public:
    GpuSharedBindings(const std::shared_ptr<compute::GPU>& gpu) : gpu_(gpu) {}

    std::optional<std::string> initialize();
    const wgpu::Texture& getTexture(uint32_t lod) const;

   private:
    friend FillPyramidPass;

    std::shared_ptr<compute::GPU> gpu_;

    std::optional<std::string> initTextures();
    std::optional<std::string> initSrcTexture(compute::Awaiter& awaiter);
    void initTexture(compute::Awaiter& awaiter, uint32_t lod);

    std::array<wgpu::Texture, GPUConst::levels_of_detail> textures_;
};
};  // namespace wslam
