#include "pose_graph.hpp"

#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Geometry>

#include "assert.hpp"

using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;

namespace {
constexpr unsigned char kNodeChar = 'n';

[[nodiscard]] gtsam::Key NodeKey(uint64_t keyframe) {
    return gtsam::Symbol(kNodeChar, keyframe);
}

[[nodiscard]] gtsam::ISAM2Params MakeIsamParams() {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    return params;
}

[[nodiscard]] gtsam::SharedNoiseModel PoseSigmas(double rotation,
                                                 double translation) {
    return gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector6() << rotation, rotation, rotation, translation,
         translation, translation)
            .finished());
}
}  // namespace

gtsam::Pose3 loop_closure::ProjectToYaw(const gtsam::Pose3& transform,
                                        const Eigen::Vector3d& axis) {
    if (axis.squaredNorm() < 0.5) {
        return transform;
    }

    // Swing-twist decomposition: the twist about axis is the quaternion with
    // the vector part projected onto the axis
    const Eigen::Quaterniond rotation = transform.rotation().toQuaternion();
    const Eigen::Vector3d along = axis * axis.dot(rotation.vec());
    Eigen::Quaterniond twist{rotation.w(), along.x(), along.y(), along.z()};
    if (twist.squaredNorm() < 1e-12) {
        // A half turn about an axis perpendicular to this one: no twist
        twist = Eigen::Quaterniond::Identity();
    }
    twist.normalize();

    return gtsam::Pose3{gtsam::Rot3{twist}, transform.translation()};
}

PoseGraph::PoseGraph(PoseGraphNoise noise, NodeGate gate)
    : noise_(noise), gate_(gate), isam_(MakeIsamParams()) {}

void PoseGraph::setGravity(const Eigen::Vector3d& gravity_world) {
    if (gravity_world.squaredNorm() > 0.0) {
        up_world_ = -gravity_world.normalized();
    }
}

bool PoseGraph::passesGate(const gtsam::Pose3& T_world_cam_vio) const {
    if (!last_node_) {
        return true;
    }
    const gtsam::Pose3 relative
        = vio_poses_.at(*last_node_).between(T_world_cam_vio);
    return relative.translation().norm() >= gate_.min_translation_m
           || relative.rotation().axisAngle().second >= gate_.min_rotation_rad;
}

bool PoseGraph::addKeyframe(uint64_t keyframe,
                            const gtsam::Pose3& T_world_cam_vio) {
    WSLAM_ASSERT(vio_poses_.empty() || keyframe > vio_poses_.rbegin()->first,
                 "keyframe {} after {}", keyframe,
                 vio_poses_.empty() ? 0 : vio_poses_.rbegin()->first);

    const bool becomes_node = passesGate(T_world_cam_vio);
    vio_poses_.emplace(keyframe, T_world_cam_vio);

    if (!becomes_node) {
        node_of_.emplace(keyframe, *last_node_);
        return false;
    }

    gtsam::NonlinearFactorGraph factors;
    gtsam::Values values;
    const gtsam::Key key = NodeKey(keyframe);

    if (!last_node_) {
        // Gauge: the first node stays where the VIO put it
        factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            key, T_world_cam_vio,
            gtsam::noiseModel::Isotropic::Sigma(6, noise_.prior_sigma));
        values.insert(key, T_world_cam_vio);
    } else {
        const uint64_t previous = *last_node_;
        const gtsam::Pose3 odometry
            = vio_poses_.at(previous).between(T_world_cam_vio);
        factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            NodeKey(previous), key, odometry,
            PoseSigmas(noise_.odometry_rotation_sigma_rad,
                       noise_.odometry_translation_sigma_m));
        // Continue from where the previous node was corrected to
        values.insert(key, estimate_.at<gtsam::Pose3>(NodeKey(previous))
                               .compose(odometry));
    }

    if (up_world_.squaredNorm() > 0.0) {
        // Measured: the up direction the VIO estimate sees in the camera
        const Eigen::Vector3d up_camera
            = T_world_cam_vio.rotation().unrotate(up_world_);
        factors.emplace_shared<gtsam::Pose3AttitudeFactor>(
            key, gtsam::Unit3{up_world_},
            gtsam::noiseModel::Isotropic::Sigma(2, noise_.attitude_sigma_rad),
            gtsam::Unit3{up_camera});
    }

    isam_.update(factors, values);
    estimate_ = isam_.calculateEstimate();

    node_of_.emplace(keyframe, keyframe);
    last_node_ = keyframe;
    ++node_count_;
    return true;
}

void PoseGraph::addLoop(const LoopEdge& edge) {
    WSLAM_ASSERT(isNode(edge.candidate) && isNode(edge.query),
                 "loop between keyframes {} and {} that are not both nodes",
                 edge.candidate, edge.query);

    gtsam::NonlinearFactorGraph factors;
    gtsam::SharedNoiseModel noise = PoseSigmas(
        noise_.loop_rotation_sigma_rad, noise_.loop_translation_sigma_m);
    if (noise_.loop_robust_k > 0.0) {
        noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(noise_.loop_robust_k),
            noise);
    }
    factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        NodeKey(edge.candidate), NodeKey(edge.query), edge.T_candidate_query,
        noise);

    isam_.update(factors, gtsam::Values{});
    for (uint32_t i = 0; i < noise_.extra_updates_after_loop; ++i) {
        isam_.update();
    }
    estimate_ = isam_.calculateEstimate();

    loops_.push_back(edge);
}

bool PoseGraph::isNode(uint64_t keyframe) const {
    const auto it = node_of_.find(keyframe);
    return it != node_of_.end() && it->second == keyframe;
}

bool PoseGraph::hasKeyframe(uint64_t keyframe) const {
    return vio_poses_.contains(keyframe);
}

const gtsam::Pose3& PoseGraph::vioPose(uint64_t keyframe) const {
    return vio_poses_.at(keyframe);
}

gtsam::Pose3 PoseGraph::correctionFor(uint64_t keyframe) const {
    uint64_t node = 0;
    if (const auto it = node_of_.find(keyframe); it != node_of_.end()) {
        node = it->second;
    } else if (last_node_) {
        node = *last_node_;
    } else {
        return gtsam::Pose3{};
    }

    return estimate_.at<gtsam::Pose3>(NodeKey(node))
        .compose(vio_poses_.at(node).inverse());
}

gtsam::Pose3 PoseGraph::correctedPose(uint64_t keyframe) const {
    return correctionFor(keyframe).compose(vio_poses_.at(keyframe));
}

gtsam::Pose3 PoseGraph::driftCorrection() const {
    if (!last_node_) {
        return gtsam::Pose3{};
    }
    return ProjectToYaw(correctionFor(*last_node_), up_world_);
}
