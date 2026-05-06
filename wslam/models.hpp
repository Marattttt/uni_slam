#pragma once

#include <array>
#include <cstdint>
namespace wslam {
struct FeatureStyle {
    std::array<uint8_t, 3> color;
    uint8_t thickness;
};

struct Feature {
    uint32_t x;
    uint32_t y;
    uint32_t strength;
};

namespace gpumodels {
template <std::size_t IMG_WIDTH, std::size_t IMG_HEIGHT>
struct FeaturesBlock {
    std::uint32_t width;
    std::uint32_t height;
    std::array<uint32_t, IMG_WIDTH * IMG_HEIGHT> strengths;

    [[nodiscard]] constexpr uint32_t operator[](std::size_t x,
                                                std::size_t y) const {
        return strengths.at(y * width + x);
    }
};
}  // namespace gpumodels
};  // namespace wslam
