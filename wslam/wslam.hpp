#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "common.hpp"
#include "compute.hpp"
#include "feature_detect.hpp"
#include "mapping.hpp"
#include "pass.hpp"
#include "pose_estimate_cpu.hpp"
#include "provider_base.hpp"

namespace wslam {

// Long-lived handles created by CreateWslamPipeline that need to outlive
// Compute (they own shared CPU state referenced by passes). The caller keeps
// this struct in scope for as long as Compute runs.
//
// `flush_async` drains any in-flight iSAM2 worker job and publishes the
// final MapSnapshot. Call it once after the main pipeline loop exits and
// before any consumer (e.g. ExportMap) reads the snapshot — otherwise the
// last frame's optimisation will be lost. Must be invoked while `compute`
// is still alive because it references the iSAM update pass owned by it.
struct WslamPipelineHandles {
    std::shared_ptr<MappingState> mapping_state;
    std::function<std::optional<std::string>()> flush_async;
};

inline WslamPipelineHandles CreateWslamPipeline(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    WslamConfig config = {}) {
    compute::Stage clearing_stage{"Clear bindings", compute};
    clearing_stage.add_pass(std::make_unique<compute::CustomPass>(
        "[Clear bindings pass]", [gpu = compute.getGPUPtr()](void*) {
            return gpu->clearBuffersAndOffsets();
        }));

    compute.addStage(std::move(clearing_stage));
    compute.addStage(CreateFeatureDetectStage(compute, shared,
                                              std::move(provider), "features",
                                              config));

    compute.addStage(
        CreatePoseEstimateCPUStage(compute, shared, "features", config));

    auto mapping = CreateMappingStage(compute, compute.getStorage(), config);
    compute.addStage(std::move(mapping.stage));

    return WslamPipelineHandles{
        .mapping_state = std::move(mapping.state),
        .flush_async = std::move(mapping.flush_async),
    };
}
};  // namespace wslam
