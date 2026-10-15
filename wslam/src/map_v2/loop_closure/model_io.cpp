#include "model_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>

#include "assert.hpp"
#include "feature_detect_v2/onnx_utils.hpp"

using namespace wslam;
using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;
namespace model = wslam::v2::raco_lightglue;

namespace {
// ImageNet statistics, RGB, of images in [0, 1]
constexpr std::array<float, 3> kImageNetMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kImageNetStd{0.229F, 0.224F, 0.225F};

constexpr float kPixelScale = 1.0F / std::numeric_limits<uint8_t>::max();

// Source pixels [begin, end) one output pixel covers along an axis, with the
// fraction of the first and last one inside it
struct Coverage {
    size_t begin;
    size_t end;
    float first_weight;
    float last_weight;
};

[[nodiscard]] std::vector<Coverage> AxisCoverage(size_t source, size_t output) {
    const double ratio
        = static_cast<double>(source) / static_cast<double>(output);
    std::vector<Coverage> coverage(output);
    for (size_t i = 0; i < output; ++i) {
        const double from = static_cast<double>(i) * ratio;
        const double to = static_cast<double>(i + 1) * ratio;
        const auto begin = static_cast<size_t>(std::floor(from));
        const auto end = std::min(source, static_cast<size_t>(std::ceil(to)));
        coverage[i] = Coverage{
            .begin = begin,
            .end = end,
            .first_weight = static_cast<float>(
                std::min(to, static_cast<double>(begin + 1)) - from),
            .last_weight = static_cast<float>(
                to - std::max(from, static_cast<double>(end - 1))),
        };
    }
    return coverage;
}

[[nodiscard]] float Weight(const Coverage& coverage, size_t index) {
    if (coverage.end - coverage.begin == 1) {
        return coverage.first_weight;
    }
    if (index == coverage.begin) {
        return coverage.first_weight;
    }
    if (index + 1 == coverage.end) {
        return coverage.last_weight;
    }
    return 1.0F;
}

// Each output pixel is the mean of the source area it covers (pixels cut by
// its border count by the fraction inside), in 8-bit units. Exact for any
// pair of sizes, up- or downscaling
[[nodiscard]] std::vector<float> ResizeArea(const PlaceImage& image,
                                            uint32_t width, uint32_t height) {
    const std::vector<Coverage> columns = AxisCoverage(image.width, width);
    const std::vector<Coverage> rows = AxisCoverage(image.height, height);

    std::vector<float> output(size_t{width} * height);
    for (uint32_t y = 0; y < height; ++y) {
        const Coverage& row = rows[y];
        for (uint32_t x = 0; x < width; ++x) {
            const Coverage& column = columns[x];
            float sum = 0.0F;
            float area = 0.0F;
            for (size_t sy = row.begin; sy < row.end; ++sy) {
                const float wy = Weight(row, sy);
                const uint8_t* source = image.pixels.data() + sy * image.width;
                for (size_t sx = column.begin; sx < column.end; ++sx) {
                    const float w = wy * Weight(column, sx);
                    sum += w * static_cast<float>(source[sx]);
                    area += w;
                }
            }
            output[size_t{y} * width + x] = sum / area;
        }
    }
    return output;
}
}  // namespace

PlaceImage PlaceImage::FromFrame(const data::FrameBW& frame) {
    PlaceImage image{
        .width = frame.width,
        .height = frame.height,
        .pixels = std::vector<uint8_t>(frame.pixels.size()),
    };
    std::ranges::transform(frame.pixels, image.pixels.begin(), [](float v) {
        const float scaled
            = std::round(v * std::numeric_limits<uint8_t>::max());
        return static_cast<uint8_t>(std::clamp(scaled, 0.0F, 255.0F));
    });
    return image;
}

std::vector<float> loop_closure::ToVprInput(const PlaceImage& image,
                                            const VprInputSpec& spec) {
    WSLAM_ASSERT(!image.empty() && spec.width > 0 && spec.height > 0,
                 "squashing an empty image or to an empty size");
    WSLAM_ASSERT(image.pixels.size() == size_t{image.width} * image.height,
                 "image of {}x{} holds {} pixels", image.width, image.height,
                 image.pixels.size());

    // Area averaging: the models' sizes are 2-3x smaller than the frame, and
    // a point-sampled or bilinear downscale would alias
    const std::vector<float> squashed
        = ResizeArea(image, spec.width, spec.height);

    const size_t plane = size_t{spec.width} * spec.height;
    std::vector<float> input(spec.elementCount());

    for (size_t channel = 0; channel < 3; ++channel) {
        const float mean = kImageNetMean.at(channel);
        const float inverse_std = 1.0F / kImageNetStd.at(channel);
        float* out = input.data() + channel * plane;

        for (size_t i = 0; i < plane; ++i) {
            out[i] = (squashed[i] * kPixelScale - mean) * inverse_std;
        }
    }

    return input;
}

std::expected<PlaceDescriptor, std::string> PlaceDescriptor::FromVprOutput(
    std::span<const float> output, size_t expected_dim) {
    if (output.size() != expected_dim) {
        return std::unexpected(
            std::format("descriptor has {} values, expected {}", output.size(),
                        expected_dim));
    }

    PlaceDescriptor descriptor{
        .values = std::vector<float>(output.begin(), output.end())};

    const double norm = std::sqrt(std::transform_reduce(
        descriptor.values.begin(), descriptor.values.end(), 0.0, std::plus{},
        [](float v) { return static_cast<double>(v) * v; }));
    if (!std::isfinite(norm) || norm < 1e-6) {
        return std::unexpected(
            std::format("descriptor norm is {}, not normalisable", norm));
    }

    const auto inverse_norm = static_cast<float>(1.0 / norm);
    for (float& v : descriptor.values) {
        v *= inverse_norm;
    }

    return descriptor;
}

float PlaceDescriptor::similarity(const PlaceDescriptor& other) const {
    WSLAM_ASSERT(values.size() == other.values.size(),
                 "comparing descriptors of {} and {} values", values.size(),
                 other.values.size());
    return std::inner_product(values.begin(), values.end(),
                              other.values.begin(), 0.0F);
}

std::vector<float> loop_closure::ToMatcherInput(const PlaceImage& image) {
    WSLAM_ASSERT(image.width == model::kSourceWidth
                     && image.height == model::kSourceHeight,
                 "matcher input must be {}x{}, got {}x{}", model::kSourceWidth,
                 model::kSourceHeight, image.width, image.height);

    constexpr size_t kPlane = size_t{model::kInputWidth} * model::kInputHeight;
    std::vector<float> input(model::kImagesBytes / sizeof(float));

    for (uint32_t y = 0; y < model::kInputHeight; ++y) {
        const uint8_t* row = image.pixels.data()
                             + size_t{y + model::kCropY} * image.width
                             + model::kCropX;
        for (uint32_t x = 0; x < model::kInputWidth; ++x) {
            const float value = static_cast<float>(row[x]) * kPixelScale;
            const size_t pixel = size_t{y} * model::kInputWidth + x;
            for (size_t channel = 0;
                 channel < static_cast<size_t>(model::kChannels); ++channel) {
                input[channel * kPlane + pixel] = value;
            }
        }
    }

    return input;
}

std::expected<PairMatches, std::string> PairMatches::FromMatcherOutput(
    std::span<const model::KeypointPosition> left_keypoints,
    std::span<const model::KeypointPosition> right_keypoints,
    std::span<const model::KeypointMatch> padded_matches,
    std::span<const float> padded_scores, int64_t keypoint_count) {
    const auto per_image = static_cast<size_t>(keypoint_count);
    if (left_keypoints.size() != per_image
        || right_keypoints.size() != per_image) {
        return std::unexpected(std::format(
            "got {} and {} keypoints, expected {} each", left_keypoints.size(),
            right_keypoints.size(), per_image));
    }

    auto parsed
        = v2::ParseMatches(padded_matches, padded_scores, keypoint_count);
    if (!parsed) {
        return std::unexpected(std::move(parsed).error());
    }

    PairMatches result;
    result.matches.reserve(parsed->matches.size());

    for (size_t i = 0; i < parsed->matches.size(); ++i) {
        const model::KeypointMatch& match = parsed->matches[i];
        const auto left_index = static_cast<size_t>(match.left_index);
        const auto right_index = static_cast<size_t>(match.right_index);
        result.matches.push_back(PairMatch{
            .left_index = static_cast<uint32_t>(left_index),
            .right_index = static_cast<uint32_t>(right_index),
            .score = parsed->scores[i],
            .left_px = model::ToSourcePixel(left_keypoints[left_index]),
            .right_px = model::ToSourcePixel(right_keypoints[right_index]),
        });
    }

    return result;
}
