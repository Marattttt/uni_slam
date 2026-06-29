#pragma once

#include <gtsam/inference/Factor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "anybag.hpp"
#include "map_common.hpp"
#include "pass.hpp"

namespace wslam::map {
class BuildFactorsPass : public compute::Pass {
   public:
    struct Params {
        // Prior noise on the very first pose. Tight to lock the gauge.
        double prior_pose_position_sigma = 1e-4;
        double prior_pose_rotation_sigma_rad = 1e-4;

        // Vision-only Between-pose factor noise. The CombinedImuFactor is
        // the primary inter-pose constraint, but iSAM2's local cliques
        // (landmark + two adjacent poses) need x-to-x information to
        // avoid singular partial Cholesky when the landmark's depth axis
        // is poorly observable. The visual BetweenFactor fills that gap.
        // The translation direction comes from the essential matrix and
        // is rescaled to the IMU-predicted metric step length before it
        // enters the measurement, so this sigma is in metres. Now that
        // the front-end geometry genuinely spans keyframe→keyframe (and
        // the rescale is metric from edge one), the edge is trusted
        // beyond mere regularisation: ~3° rotation / 0.3 m translation.
        double between_rotation_sigma_rad = 0.05;
        double between_translation_sigma = 0.3;

        // Pixel noise (sigma) for projection factors at LOD-0.
        double projection_pixel_sigma = 1.5;

        // Use Huber robust kernel on projection factors? Strongly recommended
        // — outlier observations slipping past RANSAC otherwise pull the
        // optimisation off the global minimum.
        bool use_robust_projection = true;
        // Raised from the textbook 1.345 px because IMU-initialised
        // poses in our hybrid VI-SLAM can produce reprojection residuals
        // of several pixels until iSAM converges. A 5 px threshold keeps
        // genuine outliers down-weighted while staying out of Huber
        // saturation for typical initial guesses.
        double huber_threshold_px = 5.0;

        // Initial-velocity prior on the first keyframe. The keyframe
        // gate anchors the first keyframe at motion *onset* (bootstrap
        // parallax gate), so V_0 is near zero but the platform may
        // already be a frame or two into its takeoff — hence looser
        // than a strict stationary assumption would allow.
        double prior_velocity_sigma = 0.25;  // m/s
        // Initial-bias prior on the first keyframe, centred on the bias
        // the keyframe gate measured over the stationary startup window
        // (gyro part measured directly; accel part zero). The factor
        // graph refines both from observations; this just gauges the
        // bias dimension away from a fully unobservable null space at
        // start-up.
        double prior_accel_bias_sigma = 0.1;  // m/s^2
        double prior_gyro_bias_sigma = 0.01;  // rad/s

        // Extra integration-noise added on top of the per-sample
        // covariance built from IMU noise densities. Soaks up unmodelled
        // discretisation error; 1e-8 is the textbook default.
        double integration_sigma = 1e-4;  // m/s/sqrt(s)
    };

    BuildFactorsPass(AnyBag& storage, MappingSharedBindings& shared,
                     Params params)
        : storage_(storage), shared_(shared), params_(params) {};

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    MappingSharedBindings shared_;
    const Params params_;

    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    gtsam::FactorIndices remove_indices_;
    std::vector<std::pair<size_t, LandmarkId>> smart_factor_positions_;

    std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params_;
};
};  // namespace wslam::map
