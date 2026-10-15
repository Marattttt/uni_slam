#include "wslam.hpp"

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <set>
#include <string>
#include <thread>
#include <utility>

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
#include "map/utils.hpp"
#include "map_v2/loop_closure/loop_closure_worker.hpp"
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

// Physical cores (what ONNX Runtime sizes a CPU session's pool to by
// default): distinct (package, core) pairs in sysfs, else the logical count
int PhysicalCoreCount() {
    std::set<std::pair<std::string, std::string>> cores;
    const std::filesystem::path cpus{"/sys/devices/system/cpu"};
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(cpus, error)) {
        const std::string name = entry.path().filename().string();
        if (!name.starts_with("cpu")
            || name.find_first_not_of("0123456789", 3) != std::string::npos
            || name.size() == 3) {
            continue;
        }
        std::ifstream package(entry.path() / "topology"
                              / "physical_package_id");
        std::ifstream core(entry.path() / "topology" / "core_id");
        std::string package_id;
        std::string core_id;
        if (package >> package_id && core >> core_id) {
            cores.emplace(package_id, core_id);
        }
    }
    if (!cores.empty()) {
        return static_cast<int>(cores.size());
    }
    return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
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
        .model_path = model.paths.cached.string(),
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

    // The ORT pass is chosen by the config, the adapter by Compute
    if (gpu->getComputeDevice() != config.compute_device) {
        return std::format(
            "compute was prepared for the {} but the pipeline asks for the {}",
            gpu->getComputeDevice(), config.compute_device);
    }

    auto model_options = v2::ReadModelOptionsFromEnv();
    if (!model_options) {
        return "model selection: " + std::move(model_options).error();
    }
    if (auto err = v2::ValidateModelFiles(model_options.value(),
                                          config.compute_device)) {
        return "model selection: " + std::move(err).value();
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

    // WSLAM_ORT_THREADS overrides the front end's CPU session pool size
    // (applied after the loop closure budget below)
    const int frontend_threads = map::utils::EnvInt("WSLAM_ORT_THREADS", 0);

    // Loop closure loads its models next to the front end's, on the same
    // provider; created before the stages so a missing model fails fast
    std::shared_ptr<v2::mapping::loop_closure::LoopClosureWorker> loop_closure;
    if (config.enable_mapping && config.enable_loop_closure) {
        namespace lc = v2::mapping::loop_closure;

        lc::LoopClosureConfig lc_defaults;
        lc_defaults.matcher_model_path = model_options->paths.cached;
        lc_defaults.matcher_keypoint_count = model_options->keypoint_count;
        auto lc_config = lc::ReadLoopClosureConfigFromEnv(lc_defaults);
        if (!lc_config) {
            return "loop closure config: " + std::move(lc_config).error();
        }

        auto camera = compute.getStorage().get<data::CamSensorParams>(
            ResourceIdentifier::GetCameraIntrinsicsName(0));
        if (!camera) {
            return "loop closure: camera parameters are not published yet";
        }

        // The front end's CPU session would take every physical core;
        // leave the loop closure sessions theirs
        if (config.compute_device == compute::ComputeDevice::Cpu) {
            const int budget = std::max(
                1, PhysicalCoreCount() - lc_config->intra_op_threads);
            ort->setDefaultCpuThreads(budget);
        }

        auto worker = lc::CreateLoopClosureWorker(
            *ort, lc_config.value(), camera.value(), compute.getPerf());
        if (!worker) {
            return "loop closure: " + std::move(worker).error();
        }
        loop_closure = std::move(worker).value();
    }

    if (frontend_threads > 0) {
        ort->setDefaultCpuThreads(frontend_threads);
    }

    compute.addStage(v2::CreateFeatureDetectStage(
        compute, bindings, std::move(ort), std::move(provider),
        model_options.value(), reference_advance, config,
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
        AddMappingStage(
            compute,
            v2::mapping::CreateMappingStage(compute, compute.getStorage(),
                                            config, std::move(loop_closure)),
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
