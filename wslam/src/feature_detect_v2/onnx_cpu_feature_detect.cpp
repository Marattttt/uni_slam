#include "onnx_cpu_feature_detect.hpp"

#include <spdlog/spdlog.h>

#include <format>
#include <span>
#include <utility>

#include "common.hpp"
#include "onnx_utils.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[ORT CPU Feature Detect pass]"

std::string v2::OrtCpuFeatureDetectPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::OrtCpuFeatureDetectPass::initialize() {
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

std::optional<std::string> v2::OrtCpuFeatureDetectPass::initBinding() {
    if (!session_) {
        return "session is not loaded";
    }

    const auto keypoints_shape
        = model::KeypointsShape(bindings_->getKeypointCount());

    images_.resize(model::kImagesBytes / sizeof(float));
    keypoints_.resize(model::KeypointsCount(bindings_->getKeypointCount()));

    try {
        cpu_memory_info_
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const Ort::Value& images_tensor
            = images_tensor_.emplace(Ort::Value::CreateTensor<float>(
                cpu_memory_info_, images_.data(), images_.size(),
                model::kImagesShape.data(), model::kImagesShape.size()));

        // A KeypointPosition is two floats, one row of the keypoints tensor
        const std::span keypoint_values{
            reinterpret_cast<float*>(keypoints_.data()), keypoints_.size() * 2};
        const Ort::Value& keypoints_tensor
            = keypoints_tensor_.emplace(Ort::Value::CreateTensor<float>(
                cpu_memory_info_, keypoint_values.data(),
                keypoint_values.size(), keypoints_shape.data(),
                keypoints_shape.size()));

        Ort::IoBinding& io_binding = io_binding_.emplace(*session_);

        io_binding.BindInput(model::kImagesName, images_tensor);

        // Output order here is the order GetOutputValues() returns them in
        io_binding.BindOutput(model::kKeypointsName, keypoints_tensor);
        BindDynamicOutputs(io_binding, cpu_memory_info_);
    } catch (const Ort::Exception& e) {
        return std::format("creating IO binding: {}", e.what());
    }

    return std::nullopt;
}

std::optional<std::string> v2::OrtCpuFeatureDetectPass::execute() {
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

    if (auto err
        = gpu_->readBufferInto(bindings_->getImagesBuffer(),
                               std::as_writable_bytes(std::span{images_}))) {
        return "reading images: " + err.value();
    }

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

    // Copied: the bound keypoints tensor keeps viewing keypoints_
    storage_.set(ResourceIdentifier::OrtKeypointsName, keypoints_);
    storage_.set(ResourceIdentifier::OrtKeypointsReadyName, true);

    return std::nullopt;
}
