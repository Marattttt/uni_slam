#include "feature_detect.hpp"

#include <charconv>
#include <cstdlib>
#include <format>
#include <memory>
#include <string_view>

#include "anybag.hpp"
#include "common.hpp"
#include "compute/gpu_stage.hpp"
#include "data/provider_base.hpp"
#include "fill_pyramid.hpp"
#include "load_keypoints.hpp"
#include "onnx_feature_detect.hpp"
#include "pack_model_input.hpp"
#include "preserve_pyramid.hpp"
#include "raco_lightglue.hpp"
#include "sensor_loader.hpp"

using namespace wslam;

#define LOG_ID "[Feature Detect v2 stage]"

namespace {
class StorageImageProvider {
   public:
    StorageImageProvider(AnyBag& storage) : storage_(storage) {}

    std::optional<std::span<const std::byte>> operator()() {
        const auto name = ResourceIdentifier::GetFrameName(0);

        auto frame = storage_.get<data::FrameBW>(name);
        if (!frame) {
            spdlog::warn(
                LOG_ID " could not get frame from storage by name:'{}'", name);
            return std::nullopt;
        }

        last_frame_ = std::move(frame.value());

        return std::as_bytes(std::span{last_frame_.pixels});
    }

   private:
    AnyBag& storage_;
    data::FrameBW last_frame_;
};
}  // namespace

std::expected<v2::ModelOptions, std::string> v2::ReadModelOptionsFromEnv() {
    const char* model_path = std::getenv(WSLAM_ONNX_MODEL_ENV);
    if (model_path == nullptr || model_path[0] == '\0') {
        return std::unexpected("env variable " WSLAM_ONNX_MODEL_ENV
                               " is not set; point it at the model file");
    }

    ModelOptions options{
        .model_path = model_path,
        .keypoint_count = raco_lightglue::kDefaultKeypointCount,
    };

    const char* keypoints = std::getenv(WSLAM_ONNX_KEYPOINTS_ENV);
    if (keypoints != nullptr && keypoints[0] != '\0') {
        const std::string_view text{keypoints};
        int64_t keypoint_count = 0;
        const auto [end, error] = std::from_chars(
            text.data(), text.data() + text.size(), keypoint_count);
        if (error != std::errc{} || end != text.data() + text.size()
            || keypoint_count <= 0) {
            return std::unexpected(
                std::format(WSLAM_ONNX_KEYPOINTS_ENV
                            "='{}' is not a positive integer",
                            text));
        }
        options.keypoint_count = keypoint_count;
    }

    spdlog::info(LOG_ID " model {}, {} keypoints per image",
                 options.model_path.string(), options.keypoint_count);

    return options;
}

std::unique_ptr<compute::Stage> v2::CreateFeatureDetectStage(
    compute::Compute& compute, std::shared_ptr<SharedBindings> bindings,
    std::shared_ptr<compute::OrtProvider> ort,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const std::filesystem::path& model_path, const WslamConfig& config) {
    const auto gpu = compute.getGPUPtr();
    auto stage = std::make_unique<compute::GpuStage>("Feature detect v2", gpu,
                                                     &compute.getPerf());

    stage->add_pass(std::make_unique<SensorLoaderPass>(std::move(provider),
                                                       compute.getStorage()));

    StorageImageProvider img_provider{compute.getStorage()};
    stage->add_pass(std::make_unique<FillPyramidPass>(
        gpu, bindings,
        PassFillPyramidOpts{
            .image_getter = std::move(img_provider),
            .storage = compute.getStorage(),
        }));

    stage->add_pass(std::make_unique<PackModelInputPass>(gpu, bindings));

    // CPU pass: the stage submits and waits for the packing above first
    stage->add_pass(std::make_unique<OrtFeatureDetectPass>(
        std::move(ort), model_path, bindings, compute.getStorage()));

    stage->add_pass(std::make_unique<PreservePyramidPass>(gpu, bindings));

    stage->add_pass(std::make_unique<LoadKeypointsPass>(
        gpu, std::move(bindings), compute.getStorage()));

    (void)config;

    return stage;
}
