#include "wslam.hpp"

#include <cstdlib>
#include <expected>
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
#include "frontend_recording.hpp"
#include "keyframe_schedule.hpp"
#include "map/mapping.hpp"
#include "map_v2/map_stage.hpp"
#include "pose_estimate_cpu/pose_estimate_cpu.hpp"
#include "pose_estimate_v2/pose_estimate.hpp"

using namespace wslam;

namespace {
// Adds a mapping stage and the flushables that finish it after the loop: the
// back-end runs on a worker thread a keyframe behind the main loop, so its
// last update is drained first, then the map is exported (if requested)
void AddMappingStage(compute::Compute& compute, MappingStage mapping,
                     const WslamConfig& config) {
    compute.addStage(std::move(mapping.stage));

    compute.addFlushable(
        [flush
         = std::move(mapping.flush_async)]() -> std::optional<std::string> {
            if (flush) {
                if (auto err = flush()) {
                    return "flushing mapping worker: " + std::move(err).value();
                }
            }
            std::println("Mapping worker flushed");

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

std::unique_ptr<compute::Stage> MakeCustomStage(
    compute::Compute& compute, const std::string& name,
    compute::CustomPass::Callback body) {
    auto stage = std::make_unique<compute::Stage>(name, &compute.getPerf());
    stage->add_pass(std::make_unique<compute::CustomPass>(
        std::format("[{}]", name), std::move(body)));
    return stage;
}

std::expected<v2::FrontendRecording, std::string> MakeRecordingHeader(
    const AnyBag& storage, const v2::ModelOptions& model) {
    auto camera = storage.get<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    auto imu
        = storage.get<data::IMUSensorParams>(ResourceIdentifier::ImuParamsName);
    if (!camera || !imu) {
        return std::unexpected("sensor parameters are not published yet");
    }

    const char* source = std::getenv("EUROC_DIR");
    return v2::FrontendRecording{
        .source = source != nullptr ? source : "",
        .model_path = model.model_path.string(),
        .keypoint_count = model.keypoint_count,
        .camera = std::move(camera).value(),
        .imu_params = std::move(imu).value(),
        .frames = {},
        .golden_keyframes = {},
        .golden_landmark_count = 0,
    };
}
}  // namespace

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

    AddMappingStage(compute,
                    CreateMappingStage(compute, compute.getStorage(), config),
                    config);
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

    // With mapping, the model's reference is the last accepted keyframe, so
    // the pose estimate spans the factor-graph edge
    const auto reference_advance = config.enable_mapping
                                       ? v2::ReferenceAdvance::OnKeyframeMarker
                                       : v2::ReferenceAdvance::EveryFrame;

    compute.addStage(v2::CreateFeatureDetectStage(
        compute, bindings, std::move(ort), std::move(provider),
        model_options->model_path, reference_advance, config,
        std::move(reading_filter)));

    compute.addStage(
        v2::CreatePoseEstimateStage(compute, std::move(bindings), config));

    std::shared_ptr<v2::FrontendRecorder> recorder;
    if (!config.frontend_recording_out.empty()) {
        auto header = MakeRecordingHeader(compute.getStorage(), *model_options);
        if (!header) {
            return "front-end recording: " + std::move(header).error();
        }
        recorder = std::make_shared<v2::FrontendRecorder>(
            compute.getStorage(), config.frontend_recording_out,
            std::move(header).value());
        // Before the mapping stage consumes the IMU samples
        compute.addStage(MakeCustomStage(compute, "Record front end",
                                         [recorder](compute::CustomPass*) {
                                             return recorder->captureFrame();
                                         }));
    }

    if (config.enable_mapping) {
        AddMappingStage(compute,
                        v2::mapping::CreateMappingStage(
                            compute, compute.getStorage(), config),
                        config);
    }

    if (recorder) {
        compute.addStage(MakeCustomStage(compute, "Record keyframe decision",
                                         [recorder](compute::CustomPass*) {
                                             return recorder->captureDecision();
                                         }));
        // Registered after the mapping flushables, so the final map is in
        // storage when it saves
        compute.addFlushable([recorder] { return recorder->save(); });
    }

    return std::nullopt;
}
