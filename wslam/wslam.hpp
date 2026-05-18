#pragma once

#include <memory>

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
struct WslamPipelineHandles {
    std::shared_ptr<MappingState> mapping_state;
};

inline WslamPipelineHandles CreateWslamPipeline(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    WslamConfig config = {}) {
    compute::Stage clearing_stage{"[Clear bindings]", compute.getGPUPtr()};
    clearing_stage.add_pass(std::make_unique<compute::CustomPass>(
        compute.getGPUPtr(), "[Clear bindings pass]",
        [gpu = compute.getGPUPtr()](void*) {
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
    };
}
};  // namespace wslam
