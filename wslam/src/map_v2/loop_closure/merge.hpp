#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include "loop_closure_worker.hpp"
#include "models.hpp"
#include "pose_graph.hpp"

// Export-time merge of the loop closure's pose graph into the smoother's map
namespace wslam::v2::mapping::loop_closure {

// Keyframe a landmark was first observed by, if known
using LandmarkAnchor = std::function<std::optional<uint64_t>(uint64_t)>;

struct MergeStats {
    size_t keyframes_corrected = 0;
    size_t landmarks_corrected = 0;
    // Landmarks without an anchor keyframe take the newest correction
    size_t landmarks_unanchored = 0;
};

// The VIO snapshot moved into the loop-corrected frame: every keyframe by its
// node's correction (T_corrected * T_vio^-1), every landmark by the
// correction of the keyframe that anchors it
[[nodiscard]] MapSnapshot ApplyCorrections(const MapSnapshot& vio,
                                           const PoseGraph& graph,
                                           const LandmarkAnchor& anchor_of,
                                           MergeStats* stats = nullptr);

[[nodiscard]] LoopClosureReport MakeReport(const LoopClosureWorker& worker);

}  // namespace wslam::v2::mapping::loop_closure
