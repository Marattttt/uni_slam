#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/onnx_interop.hpp"
#include "shared_bindings.hpp"

namespace wslam::v2 {

// Runs the RaCo-ALIKED-LightGlue model on the previous (left) and current
// (right) frame packed into the images buffer. Keypoints stay on the GPU in
// the keypoints buffer. Matches and their scores are data-dependent in size,
// so ORT returns them in CPU memory, and they are published to storage.
// Skips the first frame, which has no previous frame to match against
class OrtFeatureDetectPass : public compute::OrtPass {
   public:
    OrtFeatureDetectPass(std::shared_ptr<compute::OrtProvider> ort,
                         std::filesystem::path model_path,
                         std::shared_ptr<SharedBindings> bindings,
                         AnyBag& storage)
        : OrtPass(std::move(ort), std::move(model_path)),
          bindings_(std::move(bindings)),
          storage_(storage) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> submitWork() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;
    AnyBag& storage_;

    // Tensor views of the GPU-owned buffers. They don't own the buffers
    std::optional<Ort::Value> images_tensor_;
    std::optional<Ort::Value> keypoints_tensor_;
    std::optional<Ort::IoBinding> io_binding_;
    Ort::MemoryInfo cpu_memory_info_{nullptr};

    [[nodiscard]] std::optional<std::string> initBinding();
};

}  // namespace wslam::v2
