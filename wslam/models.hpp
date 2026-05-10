#pragma once

#include <array>
#include <bitset>
#include <cstdint>

#include "common.hpp"

namespace wslam {
struct FeatureStyle {
    std::array<uint8_t, 3> color;
    uint8_t thickness;
};

struct Corner {
    uint32_t x;
    uint32_t y;
    uint32_t strength;
};

// NOLINTNEXTLINE(readability-magic-numbers)
using FeatureDescriptor = std::bitset<256>;

struct Feature {
    uint32_t width;
    uint32_t height;
    uint32_t strength;
    float orientation;
    FeatureDescriptor descriptor;
};

namespace gpumodels {
// NOLINTNEXTLINE(readability-magic-numbers)
using BRIEFTestSet = std::array<int32_t, 4UZ * 256>;

template <std::size_t IMG_WIDTH, std::size_t IMG_HEIGHT>
struct CornersBlock {
    std::uint32_t width;
    std::uint32_t height;
    std::array<uint32_t, IMG_WIDTH * IMG_HEIGHT> strengths;

    [[nodiscard]] constexpr uint32_t operator[](std::size_t x,
                                                std::size_t y) const {
        return strengths.at(y * width + x);
    }
};

template <std::size_t MaxFeatures
          = GPUConst::frame_width * GPUConst::frame_height>
struct FeatureArray {
    uint32_t count;
    std::array<Feature, MaxFeatures> values;
};
}  // namespace gpumodels
};  // namespace wslam
