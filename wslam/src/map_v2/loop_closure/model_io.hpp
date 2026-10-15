#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "data/provider_base.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"

// Input and output types shared by the loop closure models, and the adapters
// between them and each model's tensors. Every model starts from a
// PlaceImage; the place recognition models end in a PlaceDescriptor, the
// matcher in PairMatches
namespace wslam::v2::mapping::loop_closure {

// Grayscale keyframe image as the camera delivered it (LOD 0, 8 bits)
struct PlaceImage {
    uint16_t width = 0;
    uint16_t height = 0;
    std::vector<uint8_t> pixels;

    // The sensor loader's frames hold the 8-bit pixels scaled to [0, 1], so
    // this conversion is exact
    [[nodiscard]] static PlaceImage FromFrame(const data::FrameBW& frame);

    [[nodiscard]] bool empty() const { return pixels.empty(); }
};

// Input of a place recognition model: float32 [1, 3, height, width], NCHW
struct VprInputSpec {
    uint32_t width = 0;
    uint32_t height = 0;

    [[nodiscard]] size_t elementCount() const {
        return size_t{3} * width * height;
    }
};

// The image squashed (aspect ratio not kept) to the model's size, the gray
// value copied to the three channels and normalised with the ImageNet mean
// and standard deviation the models were trained with (neither export
// contains the normalisation)
[[nodiscard]] std::vector<float> ToVprInput(const PlaceImage& image,
                                            const VprInputSpec& spec);

// L2-normalised global descriptor of an image
struct PlaceDescriptor {
    std::vector<float> values;

    // From a model's descriptor output. Both models already normalise; this
    // repeats it so the inner product is a cosine whatever the precision
    [[nodiscard]] static std::expected<PlaceDescriptor, std::string>
    FromVprOutput(std::span<const float> output, size_t expected_dim);

    // Cosine similarity
    [[nodiscard]] float similarity(const PlaceDescriptor& other) const;
};

// Cached RaCo-LightGlue model input for the query image: float32
// raco_lightglue::kImagesShape, RGB in [0, 1], centre-cropped like the front
// end's PackModelInputPass
[[nodiscard]] std::vector<float> ToMatcherInput(const PlaceImage& image);

// A match of the two images, in LOD-0 source pixels (observed, distorted),
// in the convention of the camera intrinsics
struct PairMatch {
    uint32_t left_index = 0;
    uint32_t right_index = 0;
    float score = 0.0F;
    Eigen::Vector2d left_px;
    Eigen::Vector2d right_px;
};

struct PairMatches {
    std::vector<PairMatch> matches;

    // From the cached model's outputs: the left (candidate) and right (query)
    // keypoints, keypoint_count each, and the zero-padded matches and scores
    // (keypoint_count rows; rows with a positive score are matches)
    [[nodiscard]] static std::expected<PairMatches, std::string>
    FromMatcherOutput(
        std::span<const raco_lightglue::KeypointPosition> left_keypoints,
        std::span<const raco_lightglue::KeypointPosition> right_keypoints,
        std::span<const raco_lightglue::KeypointMatch> padded_matches,
        std::span<const float> padded_scores, int64_t keypoint_count);
};

}  // namespace wslam::v2::mapping::loop_closure
