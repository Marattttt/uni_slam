#include "compute/onnx_interop.hpp"

#include <spdlog/spdlog.h>

#include <format>
#include <stdexcept>
#include <string>

#include "assert.hpp"

using namespace wslam::compute;

#define LOG_ID "[ORT]"

namespace {
// Name the WebGPU EP gives its device memory (onnxruntime::WEBGPU_BUFFER)
constexpr const char* kWebGpuEPBufferName = "WebGPU_Buf";

[[nodiscard]] constexpr std::optional<size_t> GetElementSize(
    ONNXTensorElementDataType element_type) {
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        default:
            if consteval {
                throw std::logic_error(
                    std::format("unsupported element type {}",
                                static_cast<int>(element_type)));
            }
            return std::nullopt;
    }
}
}  // namespace

std::expected<Ort::Session, std::string> OrtProvider::LoadModel(
    const std::filesystem::path& model_name, bool use_webgpu_ep,
    std::optional<Ort::SessionOptions> extra_opts) {
    WSLAM_ASSERT(!extra_opts, "custom session options are not implemented");

    spdlog::info(LOG_ID " loading model {} on the {} execution provider",
                 model_name.string(), use_webgpu_ep ? "WebGPU" : "CPU");

    // WebGPU EP reads the instance and device as the integer values of the raw
    // WGPUInstance / WGPUDevice handles
    const auto instance_handle
        = reinterpret_cast<size_t>(gpu_->getInstance().Get());
    const auto device_handle
        = reinterpret_cast<size_t>(gpu_->getDevice().Get());

    try {
        Ort::SessionOptions opts;
        if (!use_webgpu_ep) {
            // The CPU execution provider is the one a session always has
            return Ort::Session{*env_, model_name.c_str(), opts};
        }

        opts.AppendExecutionProvider(
            "WebGPU", {
                          {"deviceId", std::to_string(kWebGpuContextId)},
                          {"webgpuInstance", std::to_string(instance_handle)},
                          {"webgpuDevice", std::to_string(device_handle)},
                          // Preserve device after session is destroyed
                          {"preserveDevice", "1"},
                          // ORT allocates some of its buffers and this is the
                          // option for reusing memory between allocations
                          // similar to common implementations of LibC
                          {"defaultBufferCacheMode", "bucket"},
                      });

        return Ort::Session{*env_, model_name.c_str(), opts};
    } catch (const Ort::Exception& e) {
        return std::unexpected(std::format("loading model '{}': {}",
                                           model_name.string(), e.what()));
    }
}

Ort::MemoryInfo OrtProvider::getDeviceMemoryInfo() {
    // The EP reports its memory on OrtDevice id 0 whatever its context id is
    // (WebGpuDevice in vendor/onnxruntime/.../webgpu/allocator.h), so tensors
    // in the adopted device's memory use id 0, not kWebGpuContextId
    return Ort::MemoryInfo{kWebGpuEPBufferName, OrtDeviceAllocator, 0,
                           OrtMemTypeDefault};
}

std::expected<Ort::Value, std::string> OrtProvider::wrapBuffer(
    const wgpu::Buffer& buffer, ONNXTensorElementDataType element_type,
    std::span<const int64_t> shape) {
    const auto element_size = GetElementSize(element_type);
    WSLAM_ASSERT(element_size.has_value(), "unsupported element type {}",
                 static_cast<int>(element_type));

    size_t byte_count = element_size.value();
    for (const int64_t dim : shape) {
        WSLAM_ASSERT(dim >= 0, "dynamic dimension {} in a fixed tensor", dim);
        byte_count *= static_cast<size_t>(dim);
    }

    if (byte_count > buffer.GetSize()) {
        return std::unexpected(
            std::format("tensor needs {} bytes but the buffer holds {}",
                        byte_count, buffer.GetSize()));
    }

    try {
        const Ort::MemoryInfo memory_info = getDeviceMemoryInfo();
        return Ort::Value::CreateTensor(memory_info, buffer.Get(), byte_count,
                                        shape.data(), shape.size(),
                                        element_type);
    } catch (const Ort::Exception& e) {
        return std::unexpected(
            std::format("wrapping buffer as a tensor: {}", e.what()));
    }
}

std::optional<std::string> OrtPass::loadSession() {
    auto session = ort_->LoadModel(model_path_);
    if (!session) {
        return std::move(session).error();
    }

    session_ = std::move(session).value();

    return std::nullopt;
}

std::optional<std::string> OrtCpuPass::loadSession() {
    auto session = ort_->LoadModel(model_path_, /*use_webgpu_ep=*/false);
    if (!session) {
        return std::move(session).error();
    }

    session_ = std::move(session).value();

    return std::nullopt;
}
