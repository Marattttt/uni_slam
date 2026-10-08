#include "compute/onnx_interop.hpp"

#include <stdexcept>
#include <string>

using namespace wslam::compute;

std::expected<Ort::Session, std::string> OrtProvider::LoadModel(
    const std::filesystem::path& model_name,
    std::optional<Ort::SessionOptions> extra_opts) {
    Ort::SessionOptions opts;
    opts.AppendExecutionProvider(
        "WebGPU", {
                      {"webgpuInstance",
                       std::to_string(reinterpret_cast<size_t>(&device_))},
                      {"webgpuDevice",
                       std::to_string(reinterpret_cast<size_t>(&device_))},
                      {"preserveDevice", "1"},
                      {"defaultBufferCacheMode", "bucket"},
                  });

    if (extra_opts) {
        throw std::logic_error("Custom session options are not implemtented");
    }

    return Ort::Session{*env_, model_name.c_str(), opts};
}
