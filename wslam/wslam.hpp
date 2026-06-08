#pragma once

#include <memory>
#include <optional>
#include <print>
#include <string>

#include "common.hpp"
#include "compute.hpp"
#include "export.hpp"
#include "feature_detect.hpp"
#include "mapping.hpp"
#include "pass.hpp"
#include "pose_estimate_cpu.hpp"
#include "provider_base.hpp"

namespace wslam {

inline void CreateWslamPipeline(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const WslamConfig& config = {}) {
    compute::Stage clearing_stage{"Clear bindings", compute};
    clearing_stage.add_pass(std::make_unique<compute::CustomPass>(
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
        [flush_async = std::move(mapping.flush_async)]()
            -> std::optional<std::string> {
            // The iSAM2 update pass runs on a worker thread, lagging the main
            // loop by one frame. Drain any pending optimisation so the snapshot
            // under MapSnapshotName reflects every submitted keyframe before we
            // read it for export.
            if (flush_async) {
                if (auto err = flush_async()) {
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
};  // namespace wslam
