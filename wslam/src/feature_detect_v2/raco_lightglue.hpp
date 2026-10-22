#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "common.hpp"

// Contract of the split RaCo-ALIKED-LightGlue models (LightGlue-ONNX v3.0,
// `--graph extractor` and `--graph cached --pad-matches`). The extractor
// detects and describes the keypoints of one image; the cached model does the
// same for a new image and matches its keypoints against the cached ones of
// an earlier image. Model variants differ in how many keypoints they return
// per image, which is set at startup, so the keypoint-dependent parts of the
// contract are functions of that count
namespace wslam::v2::raco_lightglue {

// Tensor names
constexpr const char* kImagesName = "images";
constexpr const char* kKeypointsName = "keypoints";
constexpr const char* kDescriptorsName = "descriptors";
constexpr const char* kCachedKeypointsName = "cached_keypoints";
constexpr const char* kCachedDescriptorsName = "cached_descriptors";
constexpr const char* kMatchesName = "matches";
constexpr const char* kMatchScoresName = "mscores";

// Images per run
constexpr int64_t kBatchSize = 1;

// Image position in a matched pair. Left is the cached (earlier) image, so
// matches read as reference frame -> current frame
enum BatchSlot : uint8_t { kLeft = 0, kRight = 1 };

// RGB channels. The frames are grayscale, so the gray value is replicated
constexpr int64_t kChannels = 3;

// Length of a keypoint descriptor
constexpr int64_t kDescriptorDim = 128;

// Keypoints per image of the k512 variant. Every variant returns a fixed
// count of keypoints per image
constexpr int64_t kDefaultKeypointCount = 512;

// Model input height and width must both be multiples of this
constexpr uint32_t kSizeMultiple = 32;

// The model input is a centre crop of the camera frame, reduced to multiples
// of kSizeMultiple
constexpr uint32_t kSourceWidth = GPUConst::frame_width;
constexpr uint32_t kSourceHeight = GPUConst::frame_height;

constexpr uint32_t kInputWidth = kSourceWidth / kSizeMultiple * kSizeMultiple;
constexpr uint32_t kInputHeight = kSourceHeight / kSizeMultiple * kSizeMultiple;

// Offset of the model input inside the source level. Add it to a keypoint to
// get source-level pixel coordinates
constexpr uint32_t kCropX = (kSourceWidth - kInputWidth) / 2;
constexpr uint32_t kCropY = (kSourceHeight - kInputHeight) / 2;

static_assert(kInputWidth > 0 && kInputHeight > 0,
              "camera frame is smaller than one model tile");
static_assert(kInputWidth % kSizeMultiple == 0
                  && kInputHeight % kSizeMultiple == 0,
              "model input must be a multiple of kSizeMultiple");

// images: float32 [kBatchSize, kChannels, kInputHeight, kInputWidth], NCHW,
// RGB in [0, 1]
constexpr std::array<int64_t, 4> kImagesShape{
    kBatchSize,
    kChannels,
    kInputHeight,
    kInputWidth,
};

constexpr size_t kImagesBytes = static_cast<size_t>(kBatchSize * kChannels)
                                * kInputHeight * kInputWidth * sizeof(float);

// matches: int64 [keypoint_count, 3], one KeypointMatch per row, zero-padded
// past the real matches. mscores: float32 [keypoint_count], the score of the
// match in the same row; a row is a match only when its score is positive
constexpr int64_t kMatchesColumns = 3;

// Keypoint in model-input pixels, with the pixel-centre convention: the
// centre of the top-left pixel is (0.5, 0.5)
struct KeypointPosition {
    float x;
    float y;
};
static_assert(std::is_standard_layout_v<KeypointPosition>
              && sizeof(KeypointPosition) == 2 * sizeof(float));

// Offset from the model's pixel-centre convention to the camera intrinsics',
// where the centre of the top-left pixel is (0, 0)
constexpr double kPixelCentreOffset = 0.5;

// Keypoint position in camera pixels, in the convention of the camera
// intrinsics
[[nodiscard]] inline Eigen::Vector2d ToSourcePixel(
    const KeypointPosition& keypoint) {
    return {static_cast<double>(keypoint.x) + kCropX - kPixelCentreOffset,
            static_cast<double>(keypoint.y) + kCropY - kPixelCentreOffset};
}

// Match between a keypoint of the cached (left) image and one of the new
// (right) image. pair_index is the batch entry, always 0. Indices point into
// that image's keypoints
struct KeypointMatch {
    int64_t pair_index;
    int64_t left_index;
    int64_t right_index;
};
static_assert(std::is_standard_layout_v<KeypointMatch>
              && sizeof(KeypointMatch) == kMatchesColumns * sizeof(int64_t));

// keypoints: float32 [kBatchSize, keypoint_count, 2], one KeypointPosition
// per row
[[nodiscard]] constexpr std::array<int64_t, 3> KeypointsShape(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return {kBatchSize, keypoint_count, 2};
}

// descriptors: float32 [kBatchSize, keypoint_count, kDescriptorDim]
[[nodiscard]] constexpr std::array<int64_t, 3> DescriptorsShape(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return {kBatchSize, keypoint_count, kDescriptorDim};
}

[[nodiscard]] constexpr std::array<int64_t, 2> MatchesShape(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return {keypoint_count, kMatchesColumns};
}

[[nodiscard]] constexpr std::array<int64_t, 1> MatchScoresShape(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return {keypoint_count};
}

// Keypoints of a matched pair as published to the pose estimate: the left
// (reference) image's keypoint_count rows, then the right (current) image's
[[nodiscard]] constexpr size_t PairKeypointsCount(int64_t keypoint_count
                                                  = kDefaultKeypointCount) {
    return static_cast<size_t>(2 * keypoint_count);
}

// Keypoints and descriptors of one image: what the extractor returns, what
// the cached model takes for the earlier image and returns for the new one
struct FrameFeatures {
    // keypoint_count rows
    std::vector<KeypointPosition> keypoints;
    // keypoint_count * kDescriptorDim values, row-major
    std::vector<float> descriptors;

    [[nodiscard]] static FrameFeatures Allocate(int64_t keypoint_count) {
        const auto count = static_cast<size_t>(keypoint_count);
        return {
            .keypoints = std::vector<KeypointPosition>(count),
            .descriptors
            = std::vector<float>(count * static_cast<size_t>(kDescriptorDim)),
        };
    }

    [[nodiscard]] bool empty() const { return keypoints.empty(); }
    [[nodiscard]] int64_t keypointCount() const {
        return static_cast<int64_t>(keypoints.size());
    }
};

}  // namespace wslam::v2::raco_lightglue
