#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "compute/gpu.hpp"
#include "compute/onnx_interop.hpp"
#include "raco_lightglue.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {

// Runs the RaCo-ALIKED-LightGlue model on ONNX Runtime's CPU execution
// provider, on the previous (left) and current (right) frame packed into the
// images buffer, which it reads back. Publishes the keypoints, matches and
// their scores to storage. Skips the first frame, which has no previous frame
// to match against
class OrtCpuFeatureDetectPass : public compute::OrtCpuPass {
   public:
    OrtCpuFeatureDetectPass(std::shared_ptr<compute::OrtProvider> ort,
                            std::filesystem::path model_path,
                            std::shared_ptr<SharedBindings> bindings,
                            AnyBag& storage)
        : OrtCpuPass(std::move(ort), std::move(model_path)),
          gpu_(ort_->getGPU()),
          bindings_(std::move(bindings)),
          storage_(storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<compute::GPU> gpu_;
    std::shared_ptr<SharedBindings> bindings_;
    AnyBag& storage_;

    // Host memory the bound images and keypoints tensors view
    std::vector<float> images_;
    std::vector<raco_lightglue::KeypointPosition> keypoints_;

    std::optional<Ort::Value> images_tensor_;
    std::optional<Ort::Value> keypoints_tensor_;
    std::optional<Ort::IoBinding> io_binding_;
    Ort::MemoryInfo cpu_memory_info_{nullptr};

    [[nodiscard]] std::optional<std::string> initBinding();
};

}  // namespace wslam::v2
