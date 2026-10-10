#pragma once

#include <memory>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/stage.hpp"
#include "feature_detect_v2/shared_bindings.hpp"

namespace wslam::v2 {

// Estimates the relative pose between the previous and the current frame of
// the model's pair: loads the model's matches, filters them with RANSAC and
// triangulates the inliers. Runs after the v2 feature detect stage
std::unique_ptr<compute::Stage> CreatePoseEstimateStage(
    compute::Compute& compute, std::shared_ptr<SharedBindings> bindings,
    const WslamConfig& config = {});

}  // namespace wslam::v2
