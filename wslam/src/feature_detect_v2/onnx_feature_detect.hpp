#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "common.hpp"
#include "compute/gpu.hpp"
#include "compute/onnx_interop.hpp"

namespace wslam {
class OrtFeatureDetectPass : public compute::OrtPass {
   public:
    struct Labels {
        std::string_view keypoints;
        std::string_view matches;
        std::string_view scores;
    };
    OrtFeatureDetectPass(GpuSharedBindings& shared,
                         std::shared_ptr<compute::OrtProvider> ort,
                         const std::filesystem::path& model_path,
                         size_t num_keypoints, Labels labels)
        : OrtPass(std::move(ort), model_path),
          kNumKeypointseypoints(num_keypoints),
          labels_(labels),
          shared_(shared) {}

    [[nodiscard]] std::optional<std::string> initialize() final;
    [[nodiscard]] std::optional<std::string> execute() final;

   private:
    const size_t kNumKeypointseypoints;
    const Labels labels_;

    GpuSharedBindings& shared_;

    std::optional<Ort::Value> image_tensor_;
    std::optional<Ort::Value> keypoints_tensor_;
};
};  // namespace wslam
