#include "pose_estimate_cpu.hpp"

#include <memory>

#include "load_features_cpu.hpp"
#include "match_features_cpu.hpp"
#include "vizualize_data.hpp"

using namespace wslam;

#define LOG_ID "[Pose Estimate stage]"

compute::Stage wslam::CreatePoseEstimateCPUStage(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::string features_binding_label, WslamConfig config) {
    const auto gpu = compute.getGPUPtr();
    compute::Stage stage{"Pose Estimate", gpu};

    stage.add_pass(std::make_unique<LoadDataCPUPass>(
        shared, gpu, std::move(features_binding_label)));

    stage.add_pass(std::make_unique<MatchFeaturesCPU>(shared, gpu));

    if (config.enable_gui) {
        std::unique_ptr<viz::ResourceProvider> resource_provider
            = std::make_unique<viz::CpuResourceProvider>(
                viz::CpuResourceProvider::Opts{
                    .storage = shared.getStorage(),
                    .lod_levels = {{0}, {1}, {2}, {3}, {4}},
                    .load_features = true,
                    .load_matches = true,
                });
        stage.add_pass(std::make_unique<viz::VisualizeDataPass>(
            gpu, std::move(resource_provider)));
    }

    return stage;
}
