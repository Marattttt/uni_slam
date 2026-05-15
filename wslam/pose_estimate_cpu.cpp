#include "pose_estimate_cpu.hpp"

#include <memory>

#include "load_features_cpu.hpp"

using namespace wslam;

#define LOG_ID "[Pose Estimate stage]"

compute::Stage wslam::CreatePoseEstimateCPUStage(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::string features_binding_label) {
    const auto gpu = compute.getGPUPtr();
    compute::Stage stage{"Pose Estimate", gpu};

    stage.add_pass(std::make_unique<LoadDataCPUPass>(
        shared, gpu, "cpu:texture_data",
        std::make_pair(std::move(features_binding_label), "cpu:features")));

    return stage;
}
