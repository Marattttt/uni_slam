#include "onnx_cpu_feature_detect.hpp"

#include <spdlog/spdlog.h>

#include <format>
#include <span>

#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[ORT CPU Feature Detect pass]"

std::string v2::OrtCpuFeatureDetectPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::OrtCpuFeatureDetectPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = runner_.initialize()) {
        return "split model: " + err.value();
    }

    images_.resize(model::kImagesBytes / sizeof(float));

    try {
        const auto memory_info
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        images_tensor_.emplace(Ort::Value::CreateTensor<float>(
            memory_info, images_.data(), images_.size(),
            model::kImagesShape.data(), model::kImagesShape.size()));
    } catch (const Ort::Exception& e) {
        return std::format("creating images tensor: {}", e.what());
    }

    return std::nullopt;
}

std::optional<std::string> v2::OrtCpuFeatureDetectPass::execute() {
    if (!images_tensor_) {
        return "pass is not initialized";
    }

    spdlog::info(LOG_ID " Executing");

    if (auto err
        = gpu_->readBufferInto(bindings_->getImagesBuffer(),
                               std::as_writable_bytes(std::span{images_}))) {
        return "reading images: " + err.value();
    }

    return runner_.run(*images_tensor_);
}
