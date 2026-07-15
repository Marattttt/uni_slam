#pragma once

#include "common.hpp"
#include "compute.hpp"
#include "stage.hpp"

namespace wslam {
compute::Stage CreatePoseEstimateCPUStage(compute::Compute& compute,
                                          GpuSharedBindings& shared,
                                          std::string features_binding_label,
                                          WslamConfig config = {});
}  // namespace wslam
