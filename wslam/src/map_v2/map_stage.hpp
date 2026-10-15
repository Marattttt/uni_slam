#pragma once

#include "anybag.hpp"
#include "common.hpp"
#include "compute/compute.hpp"
#include "map/mapping.hpp"

#include <memory>

namespace wslam::v2::mapping::loop_closure {
class LoopClosureWorker;
}  // namespace wslam::v2::mapping::loop_closure

namespace wslam::v2::mapping {

// Builds the v2 mapping stage, fed by the v2 pose estimate stage and
// optimised by a sliding-window BatchFixedLagSmoother on a worker thread:
//   FilterKeyframe -> DrainSmoother -> AssociateLandmarks -> BuildFactors
//   -> SubmitSmoother
// The filter stops the stage on rejected frames, so the drain only blocks on
// accepted keyframes and the frames in between overlap the solve. Association
// and factor building run after the drain, against the window the smoother
// left. flush_async drains the last update and publishes the final snapshot.
//
// With a loop closure worker, FeedLoopClosurePass runs last, and
// flush_async also sends the worker the final drain, finishes it and merges
// its pose graph into the snapshot: MapSnapshotName then holds the
// loop-corrected map, MapSnapshotVioName the smoother's
[[nodiscard]] MappingStage CreateMappingStage(
    compute::Compute& compute, AnyBag& storage, const WslamConfig& config = {},
    std::shared_ptr<loop_closure::LoopClosureWorker> loop_closure = nullptr);

}  // namespace wslam::v2::mapping
