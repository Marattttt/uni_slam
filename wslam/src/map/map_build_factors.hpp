#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>

#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "map_common.hpp"
#include "compute/pass.hpp"

namespace wslam::map {
// Builds the per-accepted-keyframe factor-graph delta (a FactorBundle) for the
// iSAM update pass
class BuildFactorsPass : public compute::Pass {
   public:
    struct Opts {
        // Prior noise on the very first pose. Tight to lock the gauge.
        double prior_pose_position_sigma = 1e-4;
        double prior_pose_rotation_sigma_rad = 1e-4;

        // Pixel noise (sigma) for the smart projection factors at LOD-0.
        double projection_pixel_sigma = 1.5;

        // Reprojection-error threshold (in sigmas) above which the smart
        // factor's internal triangulation rejects an observation. Raised from
        // the textbook 1.345 px because IMU-initialised poses can leave
        // several-pixel residuals until iSAM converges.
        double max_observation_reprojection_px = 5.0;

        // Distance beyond which landmarks are not used for smart parameters
        double max_landmark_distance = 15.0;

        // Initial-velocity prior on the first keyframe. The keyframe gate
        // anchors the first keyframe at motion onset, so V_0 is near (but not
        // exactly) zero — hence looser than a strict stationary assumption.
        double prior_velocity_sigma = 0.25;  // m/s
        // Initial-bias prior on the first keyframe, centred on the bias the
        // filter measured over the stationary startup window (gyro measured
        // directly; accel part folded into the gravity estimate, so seeded at
        // zero). The graph refines both from observations; this just gauges the
        // bias dimension away from a fully unobservable null space at start-up.
        double prior_accel_bias_sigma = 0.1;  // m/s^2
        double prior_gyro_bias_sigma = 0.01;  // rad/s

        // Extra integration noise added on top of the per-sample covariance
        // built from the IMU noise densities. Soaks up unmodelled
        // discretisation error.
        double integration_sigma = 1e-4;
    };

    BuildFactorsPass(AnyBag& storage, MappingSharedBindings& shared,
                     Opts params)
        : storage_(storage), shared_(shared), opts_(params) {};

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    MappingSharedBindings& shared_;
    const Opts opts_;

    // Preintegration params (gravity, body_P_sensor, IMU noise) frozen once on
    // the first keyframe and reused for every CombinedImuFactor.
    std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_calibration_;

    // Set imu_calibration_ and shared_.cam_calibration if non empty
    // No-op otherwise
    //
    // @returns optional is not empty if anything needs to be set
    [[nodiscard]] std::expected<
        std::optional<std::tuple<decltype(imu_calibration_),
                                 decltype(shared_.cam_calibration)>>,
        std::string>
    computeSensorCalibration() const;

    /// Integrate IMU window in map delta.
    /// Mutates shared_.predicted_values with refined pose
    ///
    /// @param bundle factor bundle to be mutated
    /// @param delta map delta; imu range must be filled
    [[nodiscard]] std::expected<FactorBundle, std::string> integrateImu(
        FactorBundle&& bundle, const MapChanges& delta) const;

    /// First keyframe only: gauge priors on the first x / v / b plus the origin
    /// values that anchor the graph's gauge.
    [[nodiscard]] FactorBundle addGaugePriors(FactorBundle&& bundle,
                                              const MapChanges& delta) const;

    // Every keyframe: one SmartProjectionPoseFactor per observed landmark
    // (fresh, or cloned + scheduled for remove-and-readd on re-observation).
    [[nodiscard]] FactorBundle addSmartFactors(FactorBundle&& bundle,
                                               const MapChanges& delta) const;
};
};  // namespace wslam::map
