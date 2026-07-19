#pragma once

#include <memory>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/stage.hpp"

namespace wslam {
std::unique_ptr<compute::Stage> CreatePoseEstimateCPUStage(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::string features_binding_label, WslamConfig config = {});
}  // namespace wslam
