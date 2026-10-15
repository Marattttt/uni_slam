#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "compute/gpu.hpp"
#include "compute/onnx_interop.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"
#include "split_model.hpp"

namespace wslam::v2 {

// Runs the split RaCo-ALIKED-LightGlue models (SplitModelRunner) on ONNX
// Runtime's CPU execution provider, on the current frame packed into the
// images buffer, which it reads back
class OrtCpuFeatureDetectPass : public compute::Pass {
   public:
    OrtCpuFeatureDetectPass(std::shared_ptr<compute::OrtProvider> ort,
                            SplitModelPaths paths, ReferenceAdvance advance,
                            std::shared_ptr<SharedBindings> bindings,
                            AnyBag& storage)
        : gpu_(ort->getGPU()),
          bindings_(std::move(bindings)),
          runner_(std::move(ort), std::move(paths), /*use_webgpu_ep=*/false,
                  bindings_->getKeypointCount(), advance, storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<compute::GPU> gpu_;
    std::shared_ptr<SharedBindings> bindings_;
    SplitModelRunner runner_;

    // Host copy of the images buffer, viewed by images_tensor_
    std::vector<float> images_;
    std::optional<Ort::Value> images_tensor_;
};

}  // namespace wslam::v2
