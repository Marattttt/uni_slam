#include "onnx_feature_detect.hpp"

#include <spdlog/spdlog.h>

#include <format>
#include <utility>

#include "common.hpp"
#include "onnx_utils.hpp"
#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[ORT Feature Detect pass]"

std::string v2::OrtFeatureDetectPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::OrtFeatureDetectPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = loadSession()) {
        return "loading session: " + err.value();
    }

    if (auto err = ValidateModel(*session_, bindings_->getKeypointCount())) {
        return "validating model: " + err.value();
    }

    if (auto err = initBinding()) {
        return "binding tensors: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> v2::OrtFeatureDetectPass::initBinding() {
    if (!session_) {
        return "session is not loaded";
    }

    auto images = compute::OrtProvider::wrapBuffer(
        bindings_->getImagesBuffer(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        model::kImagesShape);
    if (!images) {
        return "images: " + images.error();
    }
    const Ort::Value& images_tensor
        = images_tensor_.emplace(std::move(images).value());

    auto keypoints = compute::OrtProvider::wrapBuffer(
        bindings_->getKeypointsBuffer(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        model::KeypointsShape(bindings_->getKeypointCount()));
    if (!keypoints) {
        return "keypoints: " + keypoints.error();
    }
    const Ort::Value& keypoints_tensor
        = keypoints_tensor_.emplace(std::move(keypoints).value());

    try {
        Ort::IoBinding& io_binding = io_binding_.emplace(*session_);

        io_binding.BindInput(model::kImagesName, images_tensor);

        // Output order here is the order GetOutputValues() returns them in
        io_binding.BindOutput(model::kKeypointsName, keypoints_tensor);

        cpu_memory_info_
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        BindDynamicOutputs(io_binding, cpu_memory_info_);
    } catch (const Ort::Exception& e) {
        return std::format("creating IO binding: {}", e.what());
    }

    return std::nullopt;
}

std::optional<std::string> v2::OrtFeatureDetectPass::submitWork() {
    if (!bindings_->hasPrevious()) {
        spdlog::info(LOG_ID " No previous frame to match against; skipping");
        return std::nullopt;
    }

    if (!session_ || !io_binding_) {
        return "pass is not initialized";
    }

    Ort::Session& session = *session_;
    Ort::IoBinding& io_binding = *io_binding_;

    spdlog::info(LOG_ID " Executing");

    try {
        BindDynamicOutputs(io_binding, cpu_memory_info_);
        session.Run(Ort::RunOptions{}, io_binding);

        auto outputs = io_binding.GetOutputValues();
        if (auto err = PublishMatches(outputs, bindings_->getKeypointCount(),
                                      storage_)) {
            return "publishing matches: " + err.value();
        }
    } catch (const Ort::Exception& e) {
        return std::format("running model: {}", e.what());
    }

    storage_.set(ResourceIdentifier::OrtKeypointsReadyName, true);

    return std::nullopt;
}
