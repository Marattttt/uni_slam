#pragma once

#include "common.hpp"
#include "compute.hpp"
#include "feature_detect.hpp"
#include "pose_estimate_cpu.hpp"
#include "provider_base.hpp"

namespace wslam {
constexpr void CreateWslamPipeline(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::generator<std::expected<data::Reading<1>, std::string>> provider) {
    compute.addStage(CreateFeatureDetectStage(compute, shared,
                                              std::move(provider), "features"));

    compute.addStage(CreatePoseEstimateCPUStage(compute, shared, "features"));
}
};  // namespace wslam
