#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>

#include <memory>
#include <numbers>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "map_common.hpp"

namespace wslam::v2::mapping {

// Builds the factor-graph delta of an accepted keyframe for the smoother: the
// port of v1's BuildFactorsPass, with every new key stamped with its window
// position.
//
// First keyframe: pose / velocity / bias values and their gauge priors.
// Others: a CombinedImuFactor from the previous keyframe and the initial guess
// (IMU rotation and metric step length, essential-matrix translation
// direction). Every keyframe: one SmartProjectionPoseFactor per observed
// landmark, cloned on re-observation with the old slot scheduled for removal.
//
// Input  (from AnyBag): MapChanges under MapDeltaName (taken), sensor
//                       parameters
// Output (to   AnyBag): FactorBundle under FactorBundleName, and the new
//                       keyframe's KeyframeMotionPrior under
//                       KeyframeMotionPriorName when it has an IMU edge and a
//                       scene depth
class BuildFactorsPass : public compute::Pass {
   public:
    struct Opts {
        // Prior noise on the very first pose. Tight to lock the gauge:
        // position and the rotation about gravity
        double prior_pose_position_sigma = 1e-4;
        double prior_pose_rotation_sigma_rad = 1e-4;
        // ... but not its tilt, which gravity observes: the stationary
        // window's gravity carries the accelerometer bias across it, so the
        // smoother re-levels the first pose once the body rotates. 0 locks
        // the tilt too
        double prior_pose_tilt_sigma_rad = 3.0 * std::numbers::pi / 180.0;

        // Pixel noise (sigma) for the smart projection factors at LOD-0
        double projection_pixel_sigma = 1.5;

        // Reprojection error (sigmas) above which the smart factor's
        // triangulation rejects an observation
        double max_observation_reprojection_px = 5.0;

        // Landmarks further than this are not used by the smart factors
        double max_landmark_distance = 15.0;

        // Initial-velocity prior on the first keyframe, centred on the
        // velocity integrated from the last rest. Its sigma comes from the
        // keyframe gate (SharedBindings::initial_velocity_sigma); this is the
        // fallback
        double prior_velocity_sigma = 0.25;  // m/s
        // Initial-bias prior on the first keyframe, centred on the measured
        // startup bias
        double prior_accel_bias_sigma = 0.1;  // m/s^2
        double prior_gyro_bias_sigma = 0.01;  // rad/s

        // Integration noise on top of the per-sample covariance
        double integration_sigma = 1e-4;

        // The keyframe gate's small-motion thresholds, handed to the frame
        // prefilter with the motion prior
        double gate_min_rotation_rad = 0.0;
        double gate_min_parallax_px = 0.0;

        uint32_t camera_index = 0;
    };

    BuildFactorsPass(AnyBag& storage, SharedBindings& shared, Opts opts)
        : storage_(storage), shared_(shared), opts_(opts) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

   private:
    AnyBag& storage_;
    SharedBindings& shared_;
    const Opts opts_;

    // Frozen on the first keyframe, once gravity is known
    std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_calibration_;

    // Sets imu_calibration_ and shared_.cam_calibration on first use
    [[nodiscard]] std::optional<std::string> ensureCalibration();

    void addGaugePriors(FactorBundle& bundle, const MapChanges& changes) const;
    void integrateImu(FactorBundle& bundle, const MapChanges& changes) const;
    void publishMotionPrior(const MapChanges& changes,
                            const gtsam::NavState& state,
                            double metric_step) const;
    void addSmartFactors(FactorBundle& bundle, const MapChanges& changes) const;
};

}  // namespace wslam::v2::mapping
