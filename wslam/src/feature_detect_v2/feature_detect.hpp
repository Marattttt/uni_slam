#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/onnx_interop.hpp"
#include "compute/stage.hpp"
#include "data/provider_base.hpp"
#include "shared_bindings.hpp"

// Path of the RaCo-ALIKED-LightGlue model the v2 pipeline runs. Required
#ifndef WSLAM_ONNX_MODEL_ENV
#define WSLAM_ONNX_MODEL_ENV "WSLAM_ONNX_MODEL"
#endif

// Keypoints per image the model returns. Optional, defaults to
// raco_lightglue::kDefaultKeypointCount
#ifndef WSLAM_ONNX_KEYPOINTS_ENV
#define WSLAM_ONNX_KEYPOINTS_ENV "WSLAM_ONNX_KEYPOINTS"
#endif

namespace wslam::v2 {

struct ModelOptions {
    std::filesystem::path model_path;
    int64_t keypoint_count;
};

// Reads the model selection from WSLAM_ONNX_MODEL_ENV and
// WSLAM_ONNX_KEYPOINTS_ENV
[[nodiscard]] std::expected<ModelOptions, std::string> ReadModelOptionsFromEnv();

// Detects and matches features between the previous and the current frame
// with the RaCo-ALIKED-LightGlue model
std::unique_ptr<compute::Stage> CreateFeatureDetectStage(
    compute::Compute& compute, std::shared_ptr<SharedBindings> bindings,
    std::shared_ptr<compute::OrtProvider> ort,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const std::filesystem::path& model_path, const WslamConfig& config = {});

};  // namespace wslam::v2
