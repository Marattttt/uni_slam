#pragma once

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <flat_map>
#include <memory>
#include <optional>

#include "models.hpp"
#include "data/provider_base.hpp"

namespace wslam {

// One smart factor per landmark. SmartProjectionPoseFactor marginalises
// the 3D point out of the optimisation: it stores a list of pixel
// observations + the poses that observed them, and computes a pose-only
// residual by triangulating the point internally at each linearisation.
// This sidesteps iSAM2's depth-axis singularities on low-parallax
// landmarks (which would otherwise crash partial Cholesky).
//
// We use Cal3_S2 (pinhole only) and pre-undistort observations upstream.
// Cal3DS2's inverse-distortion `calibrate()` is iterative and can fail
// to converge on edge-case pixels, which propagates as an unhandled
// exception from the smart factor's internal triangulation.
using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;

// Persistent, CPU-side state shared by every pass in the mapping stage.
//
// Holds the iSAM2 instance, the cached calibration, and the bookkeeping needed
// to associate features observed across consecutive frames with the same
// landmark variable in the factor graph.
//
// Lifetime is owned by the stage factory; passes get an `&` to the same
// instance — the same dependency-injection pattern as `GpuSharedBindings`.
struct MappingState {
    // Symbol scheme: 'x' for camera poses, 'l' for landmarks, 'v' for
    // velocity-in-world, 'b' for IMU bias (accel + gyro). Velocity and
    // bias share the keyframe's PoseId (one V_i, one B_i per accepted
    // keyframe).
    static constexpr char kPoseChar = 'x';
    static constexpr char kLandmarkChar = 'l';
    static constexpr char kVelChar = 'v';
    static constexpr char kBiasChar = 'b';

    [[nodiscard]] static gtsam::Key poseKey(PoseId id) noexcept {
        return gtsam::Symbol(kPoseChar, id.v).key();
    }
    [[nodiscard]] static gtsam::Key landmarkKey(LandmarkId id) noexcept {
        return gtsam::Symbol(kLandmarkChar, id.v).key();
    }
    [[nodiscard]] static gtsam::Key velocityKey(PoseId id) noexcept {
        return gtsam::Symbol(kVelChar, id.v).key();
    }
    [[nodiscard]] static gtsam::Key biasKey(PoseId id) noexcept {
        return gtsam::Symbol(kBiasChar, id.v).key();
    }

    // Cached pinhole calibration shared by every smart factor. Observations
    // are pre-undistorted by the keyframe gate before being added to the
    // smart factors, so distortion coefficients live separately on
    // intrinsics_cache (used for the upstream undistortion only).
    std::shared_ptr<gtsam::Cal3_S2> calibration;
    std::optional<data::CamSensorParams> intrinsics_cache;

    // Latest computed estimate. Updated on the main thread by the iSAM
    // update pass once it harvests a worker result; consumed by the
    // snapshot builder. Lags one frame behind the most recently *submitted*
    // keyframe because iSAM runs asynchronously.
    gtsam::Values latest_values;

    // Mirror of every value (pose, landmark) that has been submitted to the
    // iSAM2 worker so far, populated by the keyframe-gate pass as it builds
    // each delta. The front-end uses this — not `latest_values` — to chain
    // off the previous keyframe's pose, because at the moment the new
    // keyframe is being gated, the previous frame's iSAM update is still in
    // flight on the worker thread and has not yet written its optimised
    // estimate back into `latest_values`. Carrying the initial guess in a
    // dedicated structure keeps the front-end's pose chain decoupled from
    // back-end completion timing.
    gtsam::Values predicted_values;

    // Monotonic counters for new pose / landmark IDs.
    uint64_t next_pose_id = 0;
    uint64_t next_landmark_id = 0;

    // Did we accept at least one keyframe yet? Drives the gauge initialisation
    // logic in the factor builder.
    bool has_origin = false;
    // PoseId of the most recently accepted keyframe, if any.
    std::optional<PoseId> last_accepted_pose;

    // Maps the *current-frame* feature (output of the latest accepted
    // triangulation) to the LandmarkId it belongs to. Used by the next frame
    // to recognise re-observations: a Landmark whose `feat_prev` is in this
    // map continues an existing track.
    std::flat_map<Feature, LandmarkId> active_landmarks;

    // One smart factor per landmark. The smart factor's `add()` is called
    // on every observation; on re-observations the (mutated) factor is
    // re-added to iSAM2 via the remove-and-readd pattern below.
    std::flat_map<LandmarkId, SmartFactor::shared_ptr> smart_factors;
    // iSAM2 factor index for the most recent insertion of each smart
    // factor. Passed as `remove_factor_indices` on the next re-add so the
    // old (stale, smaller-measurement-set) version is replaced cleanly.
    // A landmark in `smart_factors` but not in this map has never been
    // added to iSAM2 yet (still pending its first update).
    std::flat_map<LandmarkId, gtsam::FactorIndex> smart_factor_indices;

    // Gravity vector expressed in the world frame (which we anchor to the
    // first accepted keyframe's camera frame). Set on the first accepted
    // keyframe from a window of stationary IMU samples; consumed by the
    // factor builder when configuring IMU preintegration. Default direction
    // is "Z+ is up" so unset gravity reads as 9.81 m/s^2 along -Z.
    Eigen::Vector3d gravity_world{0.0, 0.0, -9.81};
    bool gravity_initialised = false;

    // Timestamp (ns) of the most recently accepted keyframe. Used by the
    // keyframe gate to window IMU samples between consecutive keyframes.
    std::optional<uint64_t> last_keyframe_ts_ns;

    // PoseId -> source-frame timestamp (ns) for every accepted keyframe.
    // Populated by the keyframe gate at acceptance time and read by the
    // snapshot builder so exported KeyframePose entries carry a timestamp
    // that downstream evaluation tools can align with ground truth.
    std::flat_map<PoseId, uint64_t> keyframe_timestamps_ns;

    // Latest IMU bias estimate, propagated forward as the initial guess
    // for the next keyframe's bias variable. Filled from latest_values
    // after each iSAM update; falls back to zero on the first frame.
    gtsam::imuBias::ConstantBias last_bias;
    // Latest velocity-in-world estimate, propagated forward as the initial
    // guess for the next keyframe's velocity variable.
    Eigen::Vector3d last_velocity = Eigen::Vector3d::Zero();
};

}  // namespace wslam
