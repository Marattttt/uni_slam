#pragma once

#include "anybag.hpp"
#include "common.hpp"
#include "compute/compute.hpp"
#include "map/mapping.hpp"

namespace wslam::v2::mapping {

// Builds the v2 mapping stage, fed by the v2 pose estimate stage and
// optimised by a sliding-window BatchFixedLagSmoother on a worker thread:
//   FilterKeyframe -> DrainSmoother -> AssociateLandmarks -> BuildFactors
//   -> SubmitSmoother
// The filter stops the stage on rejected frames, so the drain only blocks on
// accepted keyframes and the frames in between overlap the solve. Association
// and factor building run after the drain, against the window the smoother
// left. flush_async drains the last update and publishes the final snapshot.
[[nodiscard]] MappingStage CreateMappingStage(compute::Compute& compute,
                                              AnyBag& storage,
                                              const WslamConfig& config = {});

}  // namespace wslam::v2::mapping
