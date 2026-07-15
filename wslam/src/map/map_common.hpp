#pragma once

#include <gtsam/base/types.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <utility>
#include <vector>

#include "models.hpp"

namespace wslam::map {
// Identifier invariant: real ids are allocated from 1; `{0}` is the reserved
// empty/none sentinel (it behaves like an optional's null state). A PoseId{0}
// therefore means "no such pose" — e.g. MapChanges::prev_pose_id == 0 marks the
// first keyframe, and last_pose / last_landmark left at {0} mean "nothing
// allocated yet".
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

[[nodiscard]] inline gtsam::Key Pose(const PoseId& id) noexcept {
    return gtsam::Symbol(kPoseCh, id.v).key();
}
[[nodiscard]] inline gtsam::Key Landmark(const LandmarkId& id) noexcept {
    return gtsam::Symbol(kLandmarkCh, id.v).key();
}
[[nodiscard]] inline gtsam::Key Velocity(const PoseId& id) noexcept {
    return gtsam::Symbol(kVelocityCh, id.v).key();
}
[[nodiscard]] inline gtsam::Key Bias(const PoseId& id) noexcept {
    return gtsam::Symbol(kBiasCh, id.v).key();
}
};  // namespace GtsamIdentifiers

// One SmartProjectionPoseFactor per landmark marginalises the 3D point out of
// the optimisation (it stores pixel observations + the poses that saw them and
// triangulates internally), sidestepping iSAM2's depth-axis singularities on
// low-parallax landmarks. Observations are pre-undistorted upstream, so the
// calibration is a pure pinhole Cal3_S2.
using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;

// Shared bindings between mapping passes
struct MappingSharedBindings {
    std::shared_ptr<gtsam::Cal3_S2> cam_calibration;

    // Count of accepted keyframes. Pose ids are allocated as
    // (keyframes_processed + 1) so the first keyframe is x1, keeping {0} as the
    // empty sentinel (see the PoseId invariant above).
    size_t keyframes_processed = 0;

    PoseId last_pose{0};
    LandmarkId last_landmark{0};
    uint64_t last_kf_ts = 0;

    // Landmarks being processed at the moment
    std::flat_map<Feature, LandmarkId> active_landmarks;

    std::flat_map<PoseId, uint64_t> pose_timestamps;

    Eigen::Vector3d gravity_world = Eigen::Vector3d::Zero();
    gtsam::imuBias::ConstantBias last_bias;

    gtsam::Values predicted_values;
    Eigen::Vector3d last_velocity = Eigen::Vector3d::Zero();

    // Persistent smart-factor bookkeeping (lives across frames).
    // `smart_factors` maps each landmark to its (mutated-on-re-observation)
    // factor; `smart_factor_indices` records the iSAM2 factor index of the most
    // recent insertion so the remove-and-readd pattern can drop the stale
    // version. A landmark present in `smart_factors` but absent from
    // `smart_factor_indices` has not been pushed to iSAM2 yet.
    std::flat_map<LandmarkId, SmartFactor::shared_ptr> smart_factors;
    std::flat_map<LandmarkId, gtsam::FactorIndex> smart_factor_indices;
};

struct Observation {
    PoseId pose;
    LandmarkId landmark;
    // Pre-undistorted LOD-0 pixel. Double to match the undistortion output and
    // the gtsam::Point2 (double) the smart factor consumes — no lossy cast.
    Eigen::Vector2d pixel;
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

// Output of BuildFactorsPass: the incremental factor-graph delta the iSAM
// update pass submits to the optimiser. Heavy GTSAM containers, moved through
// AnyBag once per accepted keyframe (ResourceIdentifier::FactorBundleName).
struct FactorBundle {
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
    // Indices of previously-inserted smart factors superseded this frame, to be
    // dropped on the next iSAM2 update (remove-and-readd).
    gtsam::FactorIndices remove_indices;
    // (position-in-new_factors, landmark) so the iSAM pass can back-fill the
    // FactorIndex iSAM2 assigns into
    // MappingSharedBindings::smart_factor_indices.
    std::vector<std::pair<size_t, LandmarkId>> smart_factor_positions;
};
};  // namespace wslam::map
