#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "anybag.hpp"
#include "compute/onnx_interop.hpp"
#include "compute/pass.hpp"
#include "cpu_model_input.hpp"
#include "split_model.hpp"

namespace wslam::v2 {

// Runs the split RaCo-ALIKED-LightGlue models (SplitModelRunner) on ONNX
// Runtime's CPU execution provider, on the host model input the v2
// SensorLoaderPass fills (CpuModelInputName)
class OrtCpuFeatureDetectPass : public compute::Pass {
   public:
    OrtCpuFeatureDetectPass(std::shared_ptr<compute::OrtProvider> ort,
                            SplitModelPaths paths, ReferenceAdvance advance,
                            int64_t keypoint_count, AnyBag& storage)
        : storage_(storage),
          runner_(std::move(ort), std::move(paths), /*use_webgpu_ep=*/false,
                  keypoint_count, advance, storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    SplitModelRunner runner_;

    // Model input in host memory, published for SensorLoaderPass
    std::shared_ptr<CpuModelInput> input_;
    // Views input_ and must be destroyed before it
    std::optional<Ort::Value> images_tensor_;
};

}  // namespace wslam::v2
