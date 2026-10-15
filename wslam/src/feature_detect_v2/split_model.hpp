#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "compute/onnx_interop.hpp"
#include "raco_lightglue.hpp"

namespace wslam::v2 {

// When the model's reference image moves on to the last frame it ran on
enum class ReferenceAdvance : uint8_t {
    // Every frame, so the model pairs consecutive frames. For runs without a
    // mapping stage
    EveryFrame,
    // Only after a frame for which the mapping stage published
    // ResourceIdentifier::FeatureReferenceAdvanceName (consumed here), so the
    // model pairs the last keyframe with the current frame
    OnKeyframeMarker,
};

// Files of the split RaCo-ALIKED-LightGlue models
struct SplitModelPaths {
    std::filesystem::path extractor;
    std::filesystem::path cached;
};

// Runs the split models on the frames of the v2 front end, whichever
// execution provider runs them. The first frame goes through the extractor,
// which is then released and replaced by the cached model; every later frame
// goes through the cached model, matched against the reference frame's
// features.
//
// Features are cached, never re-detected: the features of the last frame the
// model ran on are kept as pending, and become the reference when the frame
// does (see ReferenceAdvance). The reference keypoints are therefore exactly
// the ones the reference frame reported when it was the current frame, under
// the same indices.
//
// Per run with matches it publishes OrtMatchesName, OrtMatchScoresName,
// OrtKeypointsName (reference then current keypoints,
// raco_lightglue::PairKeypointsCount), OrtFrameFeaturesName (the current
// frame's features) and OrtKeypointsReadyName. The first frame publishes only
// OrtFrameFeaturesName.
class SplitModelRunner {
   public:
    SplitModelRunner(std::shared_ptr<compute::OrtProvider> ort,
                     SplitModelPaths paths, bool use_webgpu_ep,
                     int64_t keypoint_count, ReferenceAdvance advance,
                     AnyBag& storage)
        : ort_(std::move(ort)),
          paths_(std::move(paths)),
          use_webgpu_ep_(use_webgpu_ep),
          keypoint_count_(keypoint_count),
          advance_(advance),
          storage_(storage) {}

    // Loads and validates the extractor
    [[nodiscard]] std::optional<std::string> initialize();

    // Runs the model on the frame in images, a [1, 3, H, W] tensor in host or
    // device memory
    [[nodiscard]] std::optional<std::string> run(const Ort::Value& images);

   private:
    const std::shared_ptr<compute::OrtProvider> ort_;
    const SplitModelPaths paths_;
    const bool use_webgpu_ep_;
    const int64_t keypoint_count_;
    const ReferenceAdvance advance_;
    AnyBag& storage_;

    // Exactly one of them is loaded: the extractor until the first frame ran,
    // the cached model afterwards
    std::optional<Ort::Session> extractor_;
    std::optional<Ort::Session> cached_;

    // Shared with storage (OrtFrameFeaturesName), never mutated once filled
    std::shared_ptr<const raco_lightglue::FrameFeatures> reference_;
    std::shared_ptr<const raco_lightglue::FrameFeatures> pending_;

    // Output buffers of the cached model; matches and scores are zero-padded
    // to keypoint_count rows
    std::vector<raco_lightglue::KeypointMatch> padded_matches_;
    std::vector<float> padded_scores_;

    Ort::MemoryInfo cpu_memory_info_{nullptr};

    [[nodiscard]] std::expected<Ort::Session, std::string> loadSession(
        const std::filesystem::path& path) const;

    [[nodiscard]] std::optional<std::string> runExtractor(
        const Ort::Value& images);
    [[nodiscard]] std::optional<std::string> runCached(
        const Ort::Value& images);

    // Replaces the extractor by the cached model
    [[nodiscard]] std::optional<std::string> swapToCached();

    // Consumes the reference-advance marker and promotes the pending
    // features when the reference advances
    void advanceReference();

    // Host tensors viewing the buffers of features. ORT only writes through
    // them when they are bound as outputs
    [[nodiscard]] Ort::Value keypointsTensor(
        const raco_lightglue::FrameFeatures& features) const;
    [[nodiscard]] Ort::Value descriptorsTensor(
        const raco_lightglue::FrameFeatures& features) const;

    void publishFeatures(
        std::shared_ptr<const raco_lightglue::FrameFeatures> features);
};

}  // namespace wslam::v2
