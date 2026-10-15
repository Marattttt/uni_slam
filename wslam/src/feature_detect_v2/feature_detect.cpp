#include "feature_detect.hpp"

#include <array>
#include <charconv>
#include <cstdlib>
#include <format>
#include <memory>
#include <string>
#include <string_view>

#include "anybag.hpp"
#include "common.hpp"
#include "compute/gpu_stage.hpp"
#include "data/provider_base.hpp"
#include "fill_pyramid.hpp"
#include "frame_prefilter.hpp"
#include "onnx_cpu_feature_detect.hpp"
#include "onnx_feature_detect.hpp"
#include "pack_model_input.hpp"
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

namespace {
// Value of a required env variable naming a file
std::expected<std::filesystem::path, std::string> RequiredPath(
    const char* name, std::string_view what) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::unexpected(std::format(
            "env variable {} is not set; point it at the {}", name, what));
    }
    return std::filesystem::path{value};
}

// Name marking the CPU exports of the models
constexpr std::string_view kCpuExportMarker = "cpu";

bool IsCpuExport(const std::filesystem::path& path) {
    return path.filename().string().contains(kCpuExportMarker);
}
}  // namespace

std::expected<v2::ModelOptions, std::string> v2::ReadModelOptionsFromEnv() {
    auto extractor
        = RequiredPath(WSLAM_ONNX_EXTRACTOR_MODEL_ENV, "extractor model file");
    if (!extractor) {
        return std::unexpected(std::move(extractor).error());
    }
    auto cached = RequiredPath(WSLAM_ONNX_MODEL_ENV, "cached model file");
    if (!cached) {
        return std::unexpected(std::move(cached).error());
    }

    ModelOptions options{
        .paths = {.extractor = std::move(extractor).value(),
                  .cached = std::move(cached).value()},
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
            return std::unexpected(std::format(
                WSLAM_ONNX_KEYPOINTS_ENV "='{}' is not a positive integer",
                text));
        }
        options.keypoint_count = keypoint_count;
    }

    spdlog::info(LOG_ID
                 " extractor {}, cached model {}, {} keypoints per image",
                 options.paths.extractor.string(),
                 options.paths.cached.string(), options.keypoint_count);

    return options;
}

std::optional<std::string> v2::ValidateModelFiles(
    const ModelOptions& options, compute::ComputeDevice device) {
    const std::array models{options.paths.extractor, options.paths.cached};

    switch (device) {
        case compute::ComputeDevice::Gpu: {
            // TEMPORARY: the WebGPU exports of the split models are not made
            // yet. Remove this block once they are: the checks below and the
            // OrtFeatureDetectPass path are ready for them
            constexpr bool kWebGpuExportsProvided = false;
            if (!kWebGpuExportsProvided) {
                return std::format(
                    "--device=gpu needs WebGPU exports of the extractor and "
                    "cached models; the expected models are not provided (got "
                    "{} and {})",
                    options.paths.extractor.string(),
                    options.paths.cached.string());
            }

            for (const auto& path : models) {
                if (IsCpuExport(path)) {
                    return std::format(
                        "{} is a CPU export ('{}' in its name); --device=gpu "
                        "needs the WebGPU export",
                        path.string(), kCpuExportMarker);
                }
            }
            return std::nullopt;
        }
        case compute::ComputeDevice::Cpu:
            for (const auto& path : models) {
                if (!IsCpuExport(path)) {
                    return std::format(
                        "expected CPU exports (file name containing '{}') for "
                        "--device=cpu, got {}",
                        kCpuExportMarker, path.string());
                }
            }
            return std::nullopt;
    }

    return "unknown compute device";
}

std::unique_ptr<compute::Stage> v2::CreateFeatureDetectStage(
    compute::Compute& compute, std::shared_ptr<SharedBindings> bindings,
    std::shared_ptr<compute::OrtProvider> ort,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const ModelOptions& models, ReferenceAdvance reference_advance,
    const WslamConfig& config, data::ReadingFilter reading_filter) {
    const auto gpu = compute.getGPUPtr();
    auto stage = std::make_unique<compute::GpuStage>("Feature detect v2", gpu,
                                                     &compute.getPerf());

    stage->add_pass(std::make_unique<SensorLoaderPass>(
        std::move(provider), compute.getStorage(), std::move(reading_filter)));

    // First after the loader, so a skipped frame costs no GPU work and no
    // model run. It needs the mapping stage's keyframe prior, and a keyframe
    // schedule already picks the frames
    const bool use_prefilter = config.enable_mapping
                               && config.enable_frame_prefilter
                               && config.keyframes_path.empty();
    if (use_prefilter) {
        stage->add_pass(std::make_unique<FramePrefilterPass>(
            compute.getStorage(), ReadFramePrefilterOptionsFromEnv()));
    }

    StorageImageProvider img_provider{compute.getStorage()};
    stage->add_pass(std::make_unique<FillPyramidPass>(
        gpu, bindings,
        PassFillPyramidOpts{
            .image_getter = std::move(img_provider),
            .storage = compute.getStorage(),
        }));

    stage->add_pass(std::make_unique<PackModelInputPass>(gpu, bindings));

    switch (config.compute_device) {
        case compute::ComputeDevice::Gpu:
            // GPU pass: ORT submits to the same queue, so it runs after the
            // packing
            stage->add_pass(std::make_unique<OrtFeatureDetectPass>(
                std::move(ort), models.paths, reference_advance,
                std::move(bindings), compute.getStorage()));
            break;
        case compute::ComputeDevice::Cpu:
            // CPU pass: the stage waits for the packing before running it
            stage->add_pass(std::make_unique<OrtCpuFeatureDetectPass>(
                std::move(ort), models.paths, reference_advance,
                std::move(bindings), compute.getStorage()));
            break;
    }

    return stage;
}
