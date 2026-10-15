#include "onnx_feature_detect.hpp"

#include <spdlog/spdlog.h>

#include <utility>

#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[ORT Feature Detect pass]"

std::string v2::OrtFeatureDetectPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::OrtFeatureDetectPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = runner_.initialize()) {
        return "split model: " + err.value();
    }

    auto images = compute::OrtProvider::wrapBuffer(
        bindings_->getImagesBuffer(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        model::kImagesShape);
    if (!images) {
        return "images: " + images.error();
    }
    images_tensor_.emplace(std::move(images).value());

    return std::nullopt;
}

std::optional<std::string> v2::OrtFeatureDetectPass::submitWork() {
    if (!images_tensor_) {
        return "pass is not initialized";
    }

    spdlog::info(LOG_ID " Executing");

    return runner_.run(*images_tensor_);
}
