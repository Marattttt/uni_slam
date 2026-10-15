#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "config.hpp"

namespace wslam::v2::mapping::loop_closure {

// A verified loop: the query keyframe seen from the candidate
struct LoopEdge {
    uint64_t candidate = 0;
    uint64_t query = 0;
    // p_candidate = T_candidate_query * p_query
    gtsam::Pose3 T_candidate_query;
};

// Pose-only graph of the loop closure, on its own iSAM2: no landmarks or
// smart factors, so a loop factor moves whole trajectory segments.
//
// Keyframes arrive in id order with their VIO (smoother) pose. A keyframe
// that moved or turned enough from the last node becomes a node, linked to it
// by the relative VIO pose (odometry) and held to the VIO's roll and pitch by
// an attitude factor (gravity makes them observable to the VIO, so the graph
// only corrects yaw and position). Any other keyframe is attached to the last
// node and moves with it.
//
// The correction of a keyframe is T_corrected * T_vio^-1 of its node: the
// transform that takes its VIO pose (and the landmarks it anchors) into the
// loop-corrected frame
class PoseGraph {
   public:
    PoseGraph(PoseGraphNoise noise, NodeGate gate);

    // Gravity in the VIO world frame. Needed before the first keyframe for
    // the attitude factors; without it the graph runs without them
    void setGravity(const Eigen::Vector3d& gravity_world);

    // Adds the next keyframe (ids strictly increasing). Returns whether it
    // became a node
    bool addKeyframe(uint64_t keyframe, const gtsam::Pose3& T_world_cam_vio);

    // Adds a loop factor between two nodes and re-optimises
    void addLoop(const LoopEdge& edge);

    [[nodiscard]] bool isNode(uint64_t keyframe) const;
    [[nodiscard]] bool hasKeyframe(uint64_t keyframe) const;
    [[nodiscard]] size_t nodeCount() const { return node_count_; }
    [[nodiscard]] size_t keyframeCount() const { return vio_poses_.size(); }
    [[nodiscard]] size_t loopCount() const { return loops_.size(); }
    [[nodiscard]] std::optional<uint64_t> lastNode() const {
        return last_node_;
    }

    // VIO pose the keyframe arrived with
    [[nodiscard]] const gtsam::Pose3& vioPose(uint64_t keyframe) const;

    // Optimised pose of a node, or the corrected pose of an attached keyframe
    [[nodiscard]] gtsam::Pose3 correctedPose(uint64_t keyframe) const;

    // Correction of the keyframe's node. A keyframe the graph has not seen
    // (newer than every keyframe it got) takes the last node's; identity
    // before the first node
    [[nodiscard]] gtsam::Pose3 correctionFor(uint64_t keyframe) const;

    // The last node's correction, reduced to yaw about gravity and
    // translation: what the main loop applies to its newest poses
    [[nodiscard]] gtsam::Pose3 driftCorrection() const;

    [[nodiscard]] const std::vector<LoopEdge>& loops() const { return loops_; }

   private:
    const PoseGraphNoise noise_;
    const NodeGate gate_;

    gtsam::ISAM2 isam_;
    gtsam::Values estimate_;

    // Unit vector against gravity, VIO world frame. Zero when unknown
    Eigen::Vector3d up_world_ = Eigen::Vector3d::Zero();

    std::map<uint64_t, gtsam::Pose3> vio_poses_;
    // Node every keyframe belongs to (a node belongs to itself)
    std::map<uint64_t, uint64_t> node_of_;
    std::optional<uint64_t> last_node_;
    size_t node_count_ = 0;

    std::vector<LoopEdge> loops_;

    [[nodiscard]] bool passesGate(const gtsam::Pose3& T_world_cam_vio) const;
};

// Rotation of a transform reduced to its twist about axis (unit), keeping
// the translation: the yaw-and-position part of a drift
[[nodiscard]] gtsam::Pose3 ProjectToYaw(const gtsam::Pose3& transform,
                                        const Eigen::Vector3d& axis);

}  // namespace wslam::v2::mapping::loop_closure
