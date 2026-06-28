#pragma once

#include <gtsam/base/types.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/Values.h>

#include "models.hpp"

namespace wslam::map {
struct LandmarkId {
    size_t v;
    auto operator<=>(const LandmarkId&) const = default;
};

struct PoseId {
    size_t v;
    auto operator<=>(const PoseId&) const = default;
};

namespace GtsamIdentifiers {
constexpr char kPoseCh = 'x';
constexpr char kLandmarkCh = 'l';
constexpr char kVelocityCh = 'v';
constexpr char kBiasCh = 'b';

[[nodiscard]] gtsam::Key Pose(const PoseId& id) noexcept {
    return gtsam::Symbol(kPoseCh, id.v).key();
}
[[nodiscard]] gtsam::Key Landmark(const LandmarkId& id) noexcept {
    return gtsam::Symbol(kLandmarkCh, id.v).key();
}
[[nodiscard]] gtsam::Key Velocity(const PoseId& id) noexcept {
    return gtsam::Symbol(kVelocityCh, id.v).key();
}
[[nodiscard]] gtsam::Key Bias(const PoseId& id) noexcept {
    return gtsam::Symbol(kBiasCh, id.v).key();
}
};  // namespace GtsamIdentifiers

// Shared bindings between mapping passes
struct MappingSharedBindings {
    size_t keyframes_processed;

    PoseId last_pose;
    LandmarkId last_landmark;
    uint64_t last_kf_ts;

    // Landmarks being processed at the moment
    std::flat_map<Feature, LandmarkId> active_landmarks;

    std::flat_map<PoseId, uint64_t> pose_timestamps;

    Eigen::Vector3d gravity_world = Eigen::Vector3d::Zero();
    gtsam::imuBias::ConstantBias last_gyro_bias;

    gtsam::Values predicted_values;
};

struct Observation {
    PoseId pose;
    LandmarkId landmark;
    Eigen::Vector2f pixel;
};

struct MapChanges {
    PoseId pose_id;
    PoseId prev_pose_id;

    uint64_t prev_kf_ts;
    uint64_t curr_kf_ts;

    Eigen::Matrix3d R_world_cam = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_world_cam = Eigen::Vector3d::Zero();

    // Relative rotation between KF
    Eigen::Matrix3d R_rel = Eigen::Matrix3d::Identity();
    // Relative translation between KF
    Eigen::Vector3d t_rel = Eigen::Vector3d::Zero();

    struct Obs {
        Observation v;
        bool is_new;
    };
    // Observations of landmarks with a new flag
    std::vector<Obs> observations;

    struct NewLandmark {
        LandmarkId id;
        Eigen::Vector3d pos;
    };
    std::vector<NewLandmark> new_landmark_positions;

    // Accumulated samples between keyframes, empty on first keyframe
    std::vector<data::IMUReading> imu;

    [[nodiscard]] bool isFirstKeyframe() const { return prev_kf_ts == 0; }
};
};  // namespace wslam::map
