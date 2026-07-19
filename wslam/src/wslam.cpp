#include "wslam.hpp"

#include <memory>
#include <optional>
#include <print>
#include <string>

#include "export.hpp"
#include "feature_detect/feature_detect.hpp"
#include "map/mapping.hpp"
#include "compute/pass.hpp"
#include "pose_estimate_cpu/pose_estimate_cpu.hpp"
#include "data/provider_base.hpp"

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
