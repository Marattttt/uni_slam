#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common.hpp"

// Contract of the RaCo-ALIKED-LightGlue+ pipeline models (LightGlue-ONNX
// v3.0). Detection, description and matching run in one graph over a batch of
// image pairs. Model variants differ in how many keypoints they return per
// image, which is set at startup, so the keypoint-dependent parts of the
// contract are functions of that count
namespace wslam::v2::raco_lightglue {

// Tensor names
constexpr const char* kImagesName = "images";
constexpr const char* kKeypointsName = "keypoints";
constexpr const char* kMatchesName = "matches";
constexpr const char* kMatchScoresName = "mscores";

// Pairs of images per run. The model's batch holds the two images of each
// pair interleaved: [L0, R0, L1, R1, ...]
constexpr int64_t kPairCount = 1;
constexpr int64_t kBatchSize = 2 * kPairCount;

// Image position in a pair. Left is the earlier frame, so matches read as
// previous frame -> current frame
enum BatchSlot : uint8_t { kLeft = 0, kRight = 1 };

// RGB channels. The frames are grayscale, so the gray value is replicated
constexpr int64_t kChannels = 3;

// Keypoints per image of the k1024 variant. Every variant returns a fixed
// count of keypoints per image
constexpr int64_t kDefaultKeypointCount = 1024;

// Model input height and width must both be multiples of this
constexpr uint32_t kSizeMultiple = 32;

// Pyramid level fed to the model, centre-cropped down to multiples of
// kSizeMultiple
constexpr LOD kInputLod{0};

constexpr uint32_t kSourceWidth = static_cast<uint32_t>(
    GPUBindingSize::getPyramidLayerDimensions(kInputLod).first);
constexpr uint32_t kSourceHeight = static_cast<uint32_t>(
    GPUBindingSize::getPyramidLayerDimensions(kInputLod).second);

constexpr uint32_t kInputWidth = kSourceWidth / kSizeMultiple * kSizeMultiple;
constexpr uint32_t kInputHeight = kSourceHeight / kSizeMultiple * kSizeMultiple;

// Offset of the model input inside the source level. Add it to a keypoint to
// get source-level pixel coordinates
constexpr uint32_t kCropX = (kSourceWidth - kInputWidth) / 2;
constexpr uint32_t kCropY = (kSourceHeight - kInputHeight) / 2;

static_assert(kInputWidth > 0 && kInputHeight > 0,
              "pyramid level is smaller than one model tile");
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

// matches: int64 [M, 3], one KeypointMatch per row. M is the match count
// over the whole batch, so it is only known after a run. mscores: float32
// [M], the score of the match in the same row of matches
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

// Keypoint position in pixels of the source pyramid level (kInputLod), in the
// convention of the camera intrinsics
[[nodiscard]] inline Eigen::Vector2d ToSourcePixel(
    const KeypointPosition& keypoint) {
    return {static_cast<double>(keypoint.x) + kCropX - kPixelCentreOffset,
            static_cast<double>(keypoint.y) + kCropY - kPixelCentreOffset};
}

// Match between a keypoint of the left image and one of the right image of
// pair pair_index. Indices point into that image's keypoints
struct KeypointMatch {
    int64_t pair_index;
    int64_t left_index;
    int64_t right_index;
};
static_assert(std::is_standard_layout_v<KeypointMatch>
              && sizeof(KeypointMatch) == kMatchesColumns * sizeof(int64_t));

constexpr size_t kImagesBytes = static_cast<size_t>(kBatchSize * kChannels)
                                * kInputHeight * kInputWidth * sizeof(float);

// keypoints: float32 [kBatchSize, keypoint_count, 2], one KeypointPosition
// per row
[[nodiscard]] constexpr std::array<int64_t, 3> KeypointsShape(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return {kBatchSize, keypoint_count, 2};
}

// Keypoints over the whole batch
[[nodiscard]] constexpr size_t KeypointsCount(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return static_cast<size_t>(kBatchSize * keypoint_count);
}

[[nodiscard]] constexpr size_t KeypointsBytes(
    int64_t keypoint_count = kDefaultKeypointCount) {
    return KeypointsCount(keypoint_count) * sizeof(KeypointPosition);
}

}  // namespace wslam::v2::raco_lightglue
