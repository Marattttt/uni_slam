#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/onnx_interop.hpp"
#include "compute/pass.hpp"
#include "shared_bindings.hpp"
#include "split_model.hpp"

namespace wslam::v2 {

// Runs the split RaCo-ALIKED-LightGlue models (SplitModelRunner) on ONNX
// Runtime's WebGPU execution provider, on the current frame packed into the
// images buffer, which the model reads in place. A GPU pass: ORT submits to
// the device queue, so the run is ordered after the packing. The cached
// features, keypoints and matches are small and live in host memory
class OrtFeatureDetectPass : public compute::GPUPass {
   public:
    OrtFeatureDetectPass(std::shared_ptr<compute::OrtProvider> ort,
                         SplitModelPaths paths, ReferenceAdvance advance,
                         std::shared_ptr<SharedBindings> bindings,
                         AnyBag& storage)
        : GPUPass(ort->getGPU()),
          bindings_(std::move(bindings)),
          runner_(std::move(ort), std::move(paths), /*use_webgpu_ep=*/true,
                  bindings_->getKeypointCount(), advance, storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> submitWork() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;
    SplitModelRunner runner_;

    // Tensor view of the GPU-owned images buffer. It doesn't own the buffer
    std::optional<Ort::Value> images_tensor_;
};

}  // namespace wslam::v2
