#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/onnx_interop.hpp"
#include "compute/stage.hpp"
#include "data/provider_base.hpp"
#include "shared_bindings.hpp"
#include "split_model.hpp"

// Path of the cached RaCo-ALIKED-LightGlue model (extracts the current frame
// and matches it against the reference's cached features), which the v2
// pipeline runs on every frame after the first. Required
#ifndef WSLAM_ONNX_MODEL_ENV
#define WSLAM_ONNX_MODEL_ENV "WSLAM_ONNX_MODEL"
#endif

// Path of the RaCo-ALIKED extractor model, run once on the first frame.
// Required
#ifndef WSLAM_ONNX_EXTRACTOR_MODEL_ENV
#define WSLAM_ONNX_EXTRACTOR_MODEL_ENV "WSLAM_ONNX_EXTRACTOR_MODEL"
#endif

// Keypoints per image the model returns. Optional, defaults to
// raco_lightglue::kDefaultKeypointCount
#ifndef WSLAM_ONNX_KEYPOINTS_ENV
#define WSLAM_ONNX_KEYPOINTS_ENV "WSLAM_ONNX_KEYPOINTS"
#endif

namespace wslam::v2 {

struct ModelOptions {
    SplitModelPaths paths;
    int64_t keypoint_count;
};

// Reads the model selection from WSLAM_ONNX_EXTRACTOR_MODEL_ENV,
// WSLAM_ONNX_MODEL_ENV and WSLAM_ONNX_KEYPOINTS_ENV
[[nodiscard]] std::expected<ModelOptions, std::string>
ReadModelOptionsFromEnv();

// Checks that the model files are exports for the execution provider the
// device runs them on: CPU exports carry "cpu" in their file names (they use
// ONNX ops the WebGPU EP lacks, such as DeformConv)
[[nodiscard]] std::optional<std::string> ValidateModelFiles(
    const ModelOptions& options, compute::ComputeDevice device);

// Detects the current frame's features and matches them against the
// reference frame's cached ones with the split RaCo-ALIKED-LightGlue models.
// reference_advance selects which frames become the reference: every frame,
// or the keyframes the mapping stage accepts. reading_filter selects which
// frames and IMU samples of the provider are used
std::unique_ptr<compute::Stage> CreateFeatureDetectStage(
    compute::Compute& compute, std::shared_ptr<SharedBindings> bindings,
    std::shared_ptr<compute::OrtProvider> ort,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const ModelOptions& models, ReferenceAdvance reference_advance,
    const WslamConfig& config = {}, data::ReadingFilter reading_filter = {});

};  // namespace wslam::v2
