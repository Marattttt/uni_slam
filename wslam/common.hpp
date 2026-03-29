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

struct GPUConst {
    inline static const uint32_t frame_width = WSLAM_FRAME_WIDTH;
    inline static const uint32_t frame_height = WSLAM_FRAME_HEIGHT;
    inline static const uint32_t pyr_levels = 6;
    inline static const uint32_t pixel_size = sizeof(float);

    uint32_t frame_width_ = frame_width;
    uint32_t frame_height_ = frame_height;
    uint32_t pyr_levels_ = pyr_levels;
};

struct GPUBindingSize {
   public:
    inline static const size_t constants = sizeof(GPUConst);
    inline static const size_t source_frame
        = static_cast<size_t>(GPUConst::frame_width) * GPUConst::frame_height
          * GPUConst::pixel_size;
    inline static const size_t pyramid = []() consteval {
        uint32_t total = 0;

        uint32_t width = GPUConst::frame_width;
        uint32_t height = GPUConst::frame_height;

        for (uint32_t i = 0;
             i < GPUConst::pyr_levels && width != 1 && height != 1; i++) {
            total += width * height;
            width /= 2;
            height /= 2;
        }

        total *= sizeof(float);

        return total;
    }();
};

class GpuSharedBindings {
   public:
    wgpu::BindGroup group;
    wgpu::BindGroupLayout layout;

    // Called only once
    std::optional<std::string> initialize(
        compute::GPU& gpu,
        std::span<const std::byte, sizeof(GPUConst)> constant_data);

    [[nodiscard]] const compute::BufferBinding& getConstantsBinding() const;
    [[nodiscard]] const wgpu::Texture& getPyramid() const;

   private:
    friend FillPyramidPass;

    std::optional<std::string> initBindGroupLayout(const compute::GPU& gpu);
    std::optional<std::string> initBindGroup(compute::GPU& gpu);
    std::optional<std::string> initTexture(const compute::GPU& gpu);

    std::optional<std::string> writeConstants(
        compute::GPU& gpu,
        std::span<const std::byte, GPUBindingSize::constants> constant_data);

    std::vector<wgpu::BindGroupLayoutEntry> layout_entries_{1};
    std::vector<wgpu::BindGroupEntry> group_entries_{1};
    std::vector<std::string> labels_for_stuff_;

    wgpu::Texture frame_;

    compute::GPU::BufferBindingMap buf_bindings_;
};
};  // namespace wslam
