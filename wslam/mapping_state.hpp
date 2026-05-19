#pragma once

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <flat_map>
#include <optional>

#include "models.hpp"
#include "provider_base.hpp"

namespace wslam {

// Persistent, CPU-side state shared by every pass in the mapping stage.
//
// Holds the iSAM2 instance, the cached calibration, and the bookkeeping needed
// to associate features observed across consecutive frames with the same
// landmark variable in the factor graph.
//
// Lifetime is owned by the stage factory; passes get an `&` to the same
// instance — the same dependency-injection pattern as `GpuSharedBindings`.
struct MappingState {
    // Symbol scheme: 'x' for camera poses, 'l' for landmarks.
    static constexpr char kPoseChar = 'x';
    static constexpr char kLandmarkChar = 'l';

    [[nodiscard]] static gtsam::Key poseKey(PoseId id) noexcept {
        return gtsam::Symbol(kPoseChar, id.v).key();
    }
    [[nodiscard]] static gtsam::Key landmarkKey(LandmarkId id) noexcept {
        return gtsam::Symbol(kLandmarkChar, id.v).key();
    }

    // Cached camera calibration (Pinhole + radial-tangential distortion).
    // Populated on the first execute() once intrinsics land in AnyBag.
    boost::shared_ptr<gtsam::Cal3DS2> calibration;
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
};

}  // namespace wslam
