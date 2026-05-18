#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <flat_map>

#include "common.hpp"

namespace wslam {
struct Corner {
    uint32_t x;
    uint32_t y;
    uint32_t strength;
};

// NOLINTBEGIN(readability-magic-numbers)
struct alignas(16) Feature {
    uint32_t x;
    uint32_t y;
    uint32_t lod;
    uint32_t strength;
    float orientation;
    std::array<uint32_t, 8> descriptor;

    static constexpr size_t kDescriptorBits = 256;

    [[nodiscard]] constexpr bool getBit(size_t idx) const noexcept {
        assert(idx < kDescriptorBits);
        return ((descriptor[idx / 32] >> (idx % 32)) & 1U) != 0U;
    }

    constexpr void setBit(size_t idx, bool value) noexcept {
        assert(idx < kDescriptorBits);
        const uint32_t mask = uint32_t{1} << (idx % 32);
        if (value) {
            descriptor[idx / 32] |= mask;
        } else {
            descriptor[idx / 32] &= ~mask;
        }
    }

    constexpr auto operator<=>(const Feature&) const = default;
    constexpr bool operator==(const Feature&) const = default;
};

/*
Alignment offset checks agsinst

alias Descriptor = array<u32, 8>; // 256 bit string

struct Feature {
    coords: vec3<u32>,
    strength: u32,
    orientation: f32,
    descriptor: Descriptor,
};
*/
static_assert(sizeof(Feature) == 64);
static_assert(alignof(Feature) == 16);
static_assert(offsetof(Feature, x) == 0);
static_assert(offsetof(Feature, strength) == 12);
static_assert(offsetof(Feature, orientation) == 16);
static_assert(offsetof(Feature, descriptor) == 20);
// NOLINTEND(readability-magic-numbers)
using FeatureSet = std::array<std::vector<Feature>, GPUConst::levels_of_detail>;

using FeaturePair = std::pair<Feature, Feature>;

// Per-LOD map from current-frame feature (key) to matched prev-frame feature (value).
using MatchResult
    = std::array<std::flat_map<Feature, Feature>, GPUConst::levels_of_detail>;

// Output of the RANSAC pass — a filtered MatchResult plus geometric model and
// summary stats. The fundamental matrix is stored as a plain array so this
// header does not need to depend on Eigen.
struct RansacStats {
    size_t total_matches = 0;
    size_t inlier_count = 0;
    size_t iterations_run = 0;
    bool model_found = false;
};

struct RansacResult {
    MatchResult inliers;
    std::array<std::array<double, 3>, 3> fundamental_matrix{};
    RansacStats stats;
};

namespace gpumodels {
// NOLINTNEXTLINE(readability-magic-numbers)
using BRIEFTestSet = std::array<int32_t, 4UZ * 256>;

template <std::size_t IMG_WIDTH, std::size_t IMG_HEIGHT>
struct CornersBlock {
    uint32_t width;
    uint32_t height;
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

template <>
struct std::formatter<wslam::Corner> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static constexpr auto format(const wslam::Corner& c,
                                 std::format_context& ctx) {
        return std::format_to(ctx.out(), "{{Corner x:{} y:{} strength:{} }}",
                              c.x, c.y, c.strength);
    }
};

template <>
struct std::formatter<wslam::Feature> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static constexpr auto format(const wslam::Feature& f,
                                 std::format_context& ctx) {
        uint32_t bitcount
            = std::ranges::fold_left(f.descriptor, 0U, [](auto res, auto curr) {
                  return res + static_cast<uint32_t>(std::popcount(curr));
              });

        return std::format_to(
            ctx.out(),
            "{{Feature x:{} y:{} lod:{} strength:{} orientation:{} "
            "descriptor:{{ non-zero:{} }} }}",
            f.x, f.y, f.lod, f.strength, f.orientation, f.descriptor.size(),
            bitcount);
    }
};
