#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <webgpu/webgpu_cpp.h>

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"

namespace wslam::compute {

// Runs ONNX Runtime models on its WebGPU execution provider on the device
// owned by GPU, so tensors can live in buffers the GPU created and owns, or on
// its CPU execution provider
class OrtProvider {
   public:
    // WebGPU EP context that adopts the GPU's device. Context 0 is reserved for
    // a device ONNX Runtime creates itself
    static constexpr int kWebGpuContextId = 1;

    OrtProvider(std::shared_ptr<GPU> gpu, std::shared_ptr<Ort::Env> env)
        : gpu_(std::move(gpu)), env_(std::move(env)) {}

    [[nodiscard]] const std::shared_ptr<GPU>& getGPU() const { return gpu_; }
    [[nodiscard]] const std::shared_ptr<Ort::Env>& getEnv() const {
        return env_;
    }

    // use_webgpu_ep false leaves the session on the CPU execution provider
    [[nodiscard]] std::expected<Ort::Session, std::string> LoadModel(
        const std::filesystem::path& model_name, bool use_webgpu_ep = true,
        std::optional<Ort::SessionOptions> extra_opts = {});

    // Memory info of tensors that live in the WebGPU EP's device memory
    [[nodiscard]] static Ort::MemoryInfo getDeviceMemoryInfo();

    // Views a GPU-owned buffer as a tensor of the given type and shape. The
    // tensor does not own the buffer, which has to outlive it. ONNX Runtime
    // binds the whole buffer from offset 0, so the buffer must be a dedicated
    // one (GPU::createDedicatedBuffer), not a sub-region of a slab
    [[nodiscard]] static std::expected<Ort::Value, std::string> wrapBuffer(
        const wgpu::Buffer& buffer, ONNXTensorElementDataType element_type,
        std::span<const int64_t> shape);

   private:
    const std::shared_ptr<GPU> gpu_;
    const std::shared_ptr<Ort::Env> env_;
};

// A GPU pass: ONNX Runtime submits its own commands to the device queue and
// flushes them by the end of Session::Run, so a run is queue-ordered after the
// GPU passes before it and before the ones after it
class OrtPass : public GPUPass {
   public:
    OrtPass(std::shared_ptr<OrtProvider> ort, std::filesystem::path model_path)
        : GPUPass(ort->getGPU()),
          ort_(std::move(ort)),
          model_path_(std::move(model_path)) {}

   protected:
    // Loads model_path_ into session_
    [[nodiscard]] std::optional<std::string> loadSession();

    const std::shared_ptr<OrtProvider> ort_;
    const std::filesystem::path model_path_;

    std::optional<Ort::Session> session_;
};

// A CPU pass: runs the model on ONNX Runtime's CPU execution provider, so its
// tensors live in host memory
class OrtCpuPass : public Pass {
   public:
    OrtCpuPass(std::shared_ptr<OrtProvider> ort,
               std::filesystem::path model_path)
        : ort_(std::move(ort)), model_path_(std::move(model_path)) {}

   protected:
    // Loads model_path_ into session_
    [[nodiscard]] std::optional<std::string> loadSession();

    const std::shared_ptr<OrtProvider> ort_;
    const std::filesystem::path model_path_;

    std::optional<Ort::Session> session_;
};
};  // namespace wslam::compute
