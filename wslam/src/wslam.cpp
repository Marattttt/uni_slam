#include "wslam.hpp"

#include <format>
#include <memory>
#include <optional>
#include <print>
#include <string>

#include "compute/onnx_interop.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "export.hpp"
#include "feature_detect/feature_detect.hpp"
#include "feature_detect_v2/feature_detect.hpp"
#include "feature_detect_v2/shared_bindings.hpp"
#include "keyframe_schedule.hpp"
#include "map/mapping.hpp"
#include "pose_estimate_cpu/pose_estimate_cpu.hpp"
#include "pose_estimate_v2/pose_estimate.hpp"

using namespace wslam;

void wslam::impl::CreateWslamPipelineImpl(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const WslamConfig& config) {
    auto clearing_stage = std::make_unique<compute::Stage>("Clear bindings",
                                                           &compute.getPerf());

    clearing_stage->add_pass(std::make_unique<compute::CustomPass>(
        "[Clear bindings pass]", [gpu = compute.getGPUPtr()](void*) {
            return gpu->clearBuffersAndOffsets();
        }));

    compute.addStage(std::move(clearing_stage));
    compute.addStage(CreateFeatureDetectStage(
        compute, shared, std::move(provider), "features", config));

    compute.addStage(
        CreatePoseEstimateCPUStage(compute, shared, "features", config));

    auto mapping = CreateMappingStage(compute, compute.getStorage(), config);
    compute.addStage(std::move(mapping.stage));

    compute.addFlushable(
        [flush
         = std::move(mapping.flush_async)]() -> std::optional<std::string> {
            // The iSAM2 update pass runs on a worker thread, lagging the main
            // loop by one frame and needs to finish before using the
            // constructed map
            if (flush) {
                if (auto err = flush()) {
                    return "flushing iSAM2 worker: " + std::move(err).value();
                }
            }
            std::println("iSAM2 handle flushed");

            return std::nullopt;
        });

    compute.addFlushable([config, &compute]() -> std::optional<std::string> {
        if (!config.map_out_path.empty()) {
            std::println("[WSLAM] Begin exporting map");

            if (auto err
                = ExportMap(compute.getStorage(),
                            ExportOpts{.map_path = config.map_out_path})) {
                return "exporting map: " + std::move(err).value();
            }
        }

        return std::nullopt;
    });
}

std::optional<std::string> wslam::impl::CreateWslamPipelineV2Impl(
    compute::Compute& compute,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const WslamConfig& config) {
    const auto gpu = compute.getGPUPtr();

    auto model_options = v2::ReadModelOptionsFromEnv();
    if (!model_options) {
        return "model selection: " + std::move(model_options).error();
    }

    std::shared_ptr<Ort::Env> env;
    try {
        env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "uni_slam");
    } catch (const Ort::Exception& e) {
        return std::format("creating ONNX Runtime environment: {}", e.what());
    }

    auto ort = std::make_shared<compute::OrtProvider>(gpu, std::move(env));

    auto bindings = std::make_shared<v2::SharedBindings>(
        gpu, model_options->keypoint_count);
    if (auto err = bindings->initialize()) {
        return "v2 shared bindings: " + std::move(err).value();
    }

    data::ReadingFilter reading_filter;
    if (!config.keyframes_path.empty()) {
        auto schedule = LoadKeyframeSchedule(config.keyframes_path);
        if (!schedule) {
            return "keyframe schedule: " + std::move(schedule).error();
        }
        spdlog::info("[WSLAM] Reading only the {} keyframes of sequence '{}'",
                     schedule->frame_indices.size(), schedule->sequence);
        reading_filter = MakeKeyframeFilter(std::move(schedule).value());
    }

    compute.addStage(v2::CreateFeatureDetectStage(
        compute, bindings, std::move(ort), std::move(provider),
        model_options->model_path, config, std::move(reading_filter)));

    compute.addStage(
        v2::CreatePoseEstimateStage(compute, std::move(bindings), config));

    return std::nullopt;
}
