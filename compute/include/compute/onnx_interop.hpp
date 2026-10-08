#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <webgpu/webgpu_cpp.h>

#include "compute/gpu.hpp"
#include "compute/pass.hpp"

namespace wslam::compute {
class OrtProvider {
   public:
    OrtProvider(wgpu::Instance& instance, wgpu::Device& device,
                std::shared_ptr<Ort::Env> env)
        : instance_(instance), device_(device), env_(std::move(env)) {}

    [[nodiscard]] const wgpu::Device& GetDevice() const { return device_; }
    [[nodiscard]] const wgpu::Instance& GetInstance() const {
        return instance_;
    }
    [[nodiscard]] std::shared_ptr<Ort::Env> GetEnv() const { return env_; }

    [[nodiscard]] std::expected<Ort::Session, std::string> LoadModel(
        const std::filesystem::path& model_name,
        std::optional<Ort::SessionOptions> extra_opts = {});

   private:
    const wgpu::Instance& instance_;
    const wgpu::Device& device_;
    const std::shared_ptr<Ort::Env> env_;
};

class OrtPass : public Pass {
   public:
    OrtPass(std::shared_ptr<OrtProvider> ort, std::string model_path)
        : ort_(std::move(ort)), model_path_(std::move(model_path)) {}

   private:
    const std::shared_ptr<OrtProvider> ort_;
    const std::string model_path_;
};
};  // namespace wslam::compute
