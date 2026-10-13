#pragma once

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/FixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <cstdint>
#include <flat_map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "data/provider_base.hpp"
#include "map/map_common.hpp"

// Types shared by the passes of the v2 mapping stage. Ids, gtsam keys and the
// smart factor type are v1's (map/map_common.hpp): the factor graph is the
// same, only the front-end feeding it and the optimiser differ
namespace wslam::v2::mapping {

using wslam::map::LandmarkId;
using wslam::map::PoseId;
using wslam::map::SmartFactor;
using wslam::map::GtsamIdentifiers::Bias;
using wslam::map::GtsamIdentifiers::Landmark;
using wslam::map::GtsamIdentifiers::Pose;
using wslam::map::GtsamIdentifiers::Velocity;

// Smoother "timestamp" of a keyframe's pose / velocity / bias keys. The
// sliding window is counted in keyframes, so it is the keyframe's index
[[nodiscard]] inline double SmootherTime(const PoseId& id) {
    return static_cast<double>(id.v);
}

// A landmark seen in the last keyframe image, addressed by the model's
// keypoint index in that image
struct TrackedKeypoint {
    LandmarkId landmark;
    // LOD-0 source pixel the keypoint had when the keyframe was the current
    // frame. The model re-detects the keyframe image when it becomes the
    // reference, so a lookup by keypoint index is confirmed by this pixel
    Eigen::Vector2d pixel;
};

// Persistent state shared by the mapping passes (held by reference; owned by
// the stage factory). Touched by the main thread only: the smoother worker
// receives copies through FactorBundle, and the shared smart factors are never
// mutated after submission (they are cloned to add observations)
//
// Identifier invariant (as v1): ids start at 1, {0} means none
struct SharedBindings {
    std::shared_ptr<gtsam::Cal3_S2> cam_calibration;

    // Count of accepted keyframes. The next pose id is keyframes_processed + 1
    size_t keyframes_processed = 0;

    PoseId last_pose{0};
    LandmarkId last_landmark{0};
    uint64_t last_kf_ts = 0;

    // Landmarks observed in the last keyframe, by its keypoint index
    std::flat_map<uint32_t, TrackedKeypoint> active_landmarks;

    std::flat_map<PoseId, uint64_t> pose_timestamps;

    Eigen::Vector3d gravity_world = Eigen::Vector3d::Zero();
    gtsam::imuBias::ConstantBias last_bias;
    Eigen::Vector3d last_velocity = Eigen::Vector3d::Zero();

    // Best known pose of each keyframe still in the smoother window: the
    // optimised one once drained, the initial guess before
    gtsam::Values predicted_values;

    // Smart factor of every landmark in the window, and the smoother slot it
    // occupies once a drain has seen it inserted. A landmark in smart_factors
    // but not in smart_factor_slots has not been submitted yet
    std::flat_map<LandmarkId, SmartFactor::shared_ptr> smart_factors;
    std::flat_map<LandmarkId, size_t> smart_factor_slots;

    // Everything that left the window: the estimate of each marginalised
    // keyframe, and the last valid triangulation of each landmark whose smart
    // factor was marginalised. Append-only, input for later re-observation
    // handling outside the local smoother
    std::flat_map<PoseId, gtsam::Pose3> archived_poses;
    std::flat_map<LandmarkId, Eigen::Vector3d> archived_landmarks;
};

// Outcome of looking up a reference-image keypoint among the landmarks of the
// last keyframe
enum class TrackLookup : uint8_t {
    // The keypoint observed no landmark in the last keyframe
    Untracked,
    // It did, and the re-detected pixel agrees with the recorded one
    Tracked,
    // A landmark is recorded under the index, but at another pixel: the model
    // re-detected the keyframe image differently
    PixelMismatch,
};

struct TrackLookupResult {
    TrackLookup outcome = TrackLookup::Untracked;
    LandmarkId landmark{0};
};

[[nodiscard]] inline TrackLookupResult LookupTrack(
    const SharedBindings& shared, uint32_t prev_index,
    const Eigen::Vector2d& prev_pixel_observed, double tolerance_px) {
    const auto it = shared.active_landmarks.find(prev_index);
    if (it == shared.active_landmarks.end()) {
        return {};
    }
    if ((it->second.pixel - prev_pixel_observed).norm() > tolerance_px) {
        return {.outcome = TrackLookup::PixelMismatch};
    }
    return {.outcome = TrackLookup::Tracked, .landmark = it->second.landmark};
}

// A RANSAC inlier of the accepted keyframe, ready for landmark association
struct CandidateMatch {
    // Model keypoint indices in the reference (previous keyframe) and the
    // current image
    uint32_t prev_index;
    uint32_t curr_index;
    // Observed (distorted) LOD-0 pixels, as the model reports them
    Eigen::Vector2d prev_pixel_observed;
    Eigen::Vector2d curr_pixel_observed;
    // Undistorted pixels (K * ray), what the pinhole smart factors observe
    Eigen::Vector2d prev_pixel;
    Eigen::Vector2d curr_pixel;
};

// Keyframe accepted by FilterKeyframePass, before landmark association
struct KeyframeCandidate {
    PoseId pose_id{0};
    // {0} on the first keyframe
    PoseId prev_pose_id{0};

    uint64_t prev_kf_ts = 0;
    uint64_t curr_kf_ts = 0;

    // Initial world pose of the keyframe, chained off the previous one
    Eigen::Matrix3d R_world_cam = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_world_cam = Eigen::Vector3d::Zero();

    // p_curr = R_rel * p_prev + t_rel, unit translation
    Eigen::Matrix3d R_rel = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rel = Eigen::Vector3d::Zero();

    // Inliers with enough parallax to observe a landmark
    std::vector<CandidateMatch> matches;

    // IMU samples of (prev_kf_ts, curr_kf_ts]
    std::vector<data::IMUReading> imu;

    [[nodiscard]] bool isFirstKeyframe() const { return prev_pose_id.v == 0; }
};

struct Observation {
    PoseId pose;
    LandmarkId landmark;
    // Undistorted LOD-0 pixel
    Eigen::Vector2d pixel;
};

// Keyframe after landmark association, input of BuildFactorsPass
struct MapChanges {
    PoseId pose_id{0};
    PoseId prev_pose_id{0};

    uint64_t prev_kf_ts = 0;
    uint64_t curr_kf_ts = 0;

    Eigen::Matrix3d R_world_cam = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_world_cam = Eigen::Vector3d::Zero();

    Eigen::Matrix3d R_rel = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rel = Eigen::Vector3d::Zero();

    std::vector<Observation> observations;

    std::vector<data::IMUReading> imu;

    [[nodiscard]] bool isFirstKeyframe() const { return prev_pose_id.v == 0; }
};

// Per-keyframe delta for the smoother, built by BuildFactorsPass
struct FactorBundle {
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
    // Window position (SmootherTime) of every new key
    gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
    // Slots of smart factors superseded by a re-observation clone
    std::set<size_t> remove_slots;
};

}  // namespace wslam::v2::mapping
