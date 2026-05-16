#pragma once

#include <webgpu/webgpu_cpp.h>

#include <string>

#include "anybag.hpp"
#include "gpu.hpp"

#ifndef WSLAM_FRAME_WIDTH
#define WSLAM_FRAME_WIDTH 752  // Euroc mav
#endif

#ifndef WSLAM_FRAME_HEIGHT
#define WSLAM_FRAME_HEIGHT 480  // Euroc mav
#endif

namespace wslam {

struct LOD {
    uint8_t v;
    operator uint8_t() const { return v; }
};

namespace GPUConst {
constexpr uint32_t frame_width = WSLAM_FRAME_WIDTH;
constexpr uint32_t frame_height = WSLAM_FRAME_HEIGHT;
constexpr uint32_t pixel_size = sizeof(float);
constexpr uint32_t levels_of_detail = 6;
constexpr uint32_t min_uniform_buffer_alignment = 256;

constexpr uint32_t featuesets_stored = 2;

constexpr double lod_scale_factor = 1.2;
};  // namespace GPUConst

[[nodiscard]] constexpr uint32_t CeilDiv(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

[[nodiscard]] constexpr auto AddPadding(auto num, size_t alignment) {
    alignment -= 1;
    return (num + alignment) & ~alignment;
}

template <typename T,
          std::size_t Alignment = GPUConst::min_uniform_buffer_alignment>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] std::vector<std::byte> AddPadding(std::span<const T> data) {
    constexpr std::size_t stride
        = (sizeof(T) + Alignment - 1) / Alignment * Alignment;

    std::vector<std::byte> result(stride * data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto src = std::as_bytes(data.subspan(i, 1));
        std::ranges::copy(src, result.begin() + i * stride);
    }
    return result;
}

namespace GPUBindingSize {
constexpr std::pair<size_t, size_t> getPyramidLayerDimensions(const LOD lod) {
    if (lod.v >= GPUConst::levels_of_detail) {
        throw std::out_of_range("LoD out of range");
    }

    double scale = 1.0;

    for (uint8_t i = 0; i < lod.v; ++i) {
        scale *= GPUConst::lod_scale_factor;
    }

    const auto width = static_cast<size_t>(GPUConst::frame_width / scale);
    const auto height = static_cast<size_t>(GPUConst::frame_height / scale);
    return {width, height};
}

constexpr size_t getPyramidLayerSize(const LOD lod) {
    const auto [width, height] = getPyramidLayerDimensions(lod);
    return width * height * GPUConst::pixel_size;
}

// Size (in bytes) of the initial frame
constexpr size_t source_frame = static_cast<size_t>(GPUConst::frame_width)
                                * GPUConst::frame_height * GPUConst::pixel_size;

// Maximum space a LoD pyramid can take up
constexpr size_t max_pyramid_size = std::invoke([]() consteval {
    size_t total = 0;
    for (uint8_t i = 0; i < GPUConst::levels_of_detail; ++i) {
        total += getPyramidLayerSize(LOD{i});
    }
    return total;
});
}  // namespace GPUBindingSize

namespace ResourceIdentifier {
constexpr std::string GetFrameName(std::pair<uint32_t, LOD> info) {
    return std::format("res:frame:{}:lod:{}", info.first, info.second.v);
}
constexpr std::string GetFrameName(uint32_t frame_idx) {
    return GetFrameName({frame_idx, {0}});
};
constexpr std::string GetTextureName(size_t lod) {
    return std::format("res:texture:lod:{}", lod);
}
constexpr std::string GetImuVecName() { return "res:imu:vec"; }
constexpr std::string GetVizResourceName() { return "res:viz"; }
constexpr std::string GetProcessedFrameName(uint32_t keyframes_ago,
                                            uint32_t lod) {
    return std::format("gen:frame:-{}:lod:{}", keyframes_ago, lod);
}
constexpr std::string GetFeatureSetName(size_t keyframes_ago) {
    return std::format("gen:featureset:-{}", keyframes_ago);
}
}  // namespace ResourceIdentifier

class FillPyramidPass;

class GpuSharedBindings {
   public:
    GpuSharedBindings(const std::shared_ptr<compute::GPU>& gpu, AnyBag& storage)
        : gpu_(gpu), storage(storage) {}

    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] constexpr wgpu::Texture getTexture(size_t lod) const {
        return textures_.at(lod);
    }
    [[nodiscard]] constexpr const AnyBag& getStorage() const { return storage; }
    [[nodiscard]] constexpr AnyBag& getStorage() { return storage; }

   private:
    friend FillPyramidPass;

    std::shared_ptr<compute::GPU> gpu_;
    AnyBag& storage;

    std::optional<std::string> initTextures();
    std::optional<std::string> initSrcTexture(compute::Awaiter& awaiter);
    void initTexture(compute::Awaiter& awaiter, uint32_t lod);

    std::array<wgpu::Texture, GPUConst::levels_of_detail> textures_;
};
};  // namespace wslam
