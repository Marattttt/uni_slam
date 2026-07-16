#include "pose_estimate_cpu.hpp"

#include <memory>

#include "load_features_cpu.hpp"
#include "match_features_cpu.hpp"
#include "ransac_cpu.hpp"
#include "triangulate_cpu.hpp"
#include "viz/vizualize_data.hpp"

using namespace wslam;

#define LOG_ID "[Pose Estimate stage]"

compute::Stage wslam::CreatePoseEstimateCPUStage(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::string features_binding_label, WslamConfig config) {
    const auto gpu = compute.getGPUPtr();
    compute::Stage stage{"Pose Estimate", compute};

    // The LoD texture readback only feeds the GUI resource provider below;
    // headless runs skip it.
    stage.add_pass(std::make_unique<LoadDataCPUPass>(
        shared, gpu, std::move(features_binding_label),
        /*readback_textures=*/config.enable_gui));

    stage.add_pass(std::make_unique<MatchFeaturesCPU>(shared));

    stage.add_pass(std::make_unique<RansacCPU>(shared));

    stage.add_pass(std::make_unique<TriangulateCPU>(shared));

    if (config.enable_gui) {
        std::unique_ptr<viz::ResourceProvider> resource_provider
            = std::make_unique<viz::CpuResourceProvider>(
                viz::CpuResourceProvider::Opts{
                    .storage = shared.getStorage(),
                    .lod_levels = {{0}, {1}, {2}, {3}, {4}},
                    .load_features = true,
                    .load_matches = true,
                    .load_ransac_inliers = true,
                    .load_landmarks = true,
                });
        stage.add_pass(std::make_unique<viz::VisualizeDataPass>(
            gpu, std::move(resource_provider)));
    }

    return stage;
}
