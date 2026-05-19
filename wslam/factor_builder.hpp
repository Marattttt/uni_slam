#pragma once

#include <gtsam/inference/Key.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <memory>
#include <utility>
#include <vector>

#include "anybag.hpp"
#include "mapping_state.hpp"
#include "models.hpp"
#include "pass.hpp"

namespace wslam {

// Second pass of the mapping stage.
//
// Reads the `MapDelta` published by the keyframe-gate pass and assembles the
// graph + values payload that the iSAM2 update pass will consume:
//   - PriorFactor on the first pose (gauge fix)
//   - PriorFactor on the first landmark (scale fix) so monocular SLAM is fully
//     constrained — without it iSAM2 collapses the landmarks toward the
//     camera, since translation has unit norm by construction
//   - BetweenFactor<Pose3> for each pose-to-pose odometry edge
//   - GenericProjectionFactor<Pose3, Point3, Cal3DS2> for every observation
//
// The pass writes its output through the per-frame buffers stored on the
// MappingState rather than through AnyBag — these are heavy GTSAM containers
// and only the next pass needs them.
class FactorBuilderPass : public compute::Pass {
   public:
    struct Opts {
        // Prior noise on the very first pose. Tight to lock the gauge.
        double prior_pose_position_sigma = 1e-4;
        double prior_pose_rotation_sigma_rad = 1e-4;

        // Vision-only Between-pose factor noise. The CombinedImuFactor is
        // the primary inter-pose constraint, but iSAM2's local cliques
        // (landmark + two adjacent poses) need x-to-x information to
        // avoid singular partial Cholesky when the landmark's depth axis
        // is poorly observable. The visual BetweenFactor fills that gap.
        // Sigmas are LOOSE — the IMU factor is the real source of truth
        // for metric scale; this just regularises local conditioning.
        // Translation is unit-norm from the essential matrix so its
        // sigma is dimensionless relative to the visual chain.
        double between_rotation_sigma_rad = 0.1;
        double between_translation_sigma = 1.0;

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

        // Initial-velocity prior on the first keyframe. We assume the
        // sequence starts approximately stationary; a tight prior locks
        // V_0 ≈ 0 so the IMU has something to integrate against. Loosen
        // if the dataset is known to start mid-motion.
        double prior_velocity_sigma = 0.1;  // m/s
        // Initial-bias prior on the first keyframe. The factor graph will
        // refine both accel and gyro bias from observations; this just
        // gauges the bias dimension away from a fully unobservable null
        // space at start-up.
        double prior_accel_bias_sigma = 0.1;   // m/s^2
        double prior_gyro_bias_sigma = 0.01;   // rad/s

        // Extra integration-noise added on top of the per-sample
        // covariance built from IMU noise densities. Soaks up unmodelled
        // discretisation error; 1e-8 is the textbook default.
        double integration_sigma = 1e-4;  // m/s/sqrt(s)
    };

    FactorBuilderPass(MappingState& state, std::shared_ptr<compute::GPU> gpu,
                      Opts opts);
    FactorBuilderPass(MappingState& state, std::shared_ptr<compute::GPU> gpu);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    // Per-frame outputs consumed by the iSAM2 pass. Cleared at the top of
    // every execute().
    [[nodiscard]] const gtsam::NonlinearFactorGraph& newFactors() const {
        return new_factors_;
    }
    [[nodiscard]] const gtsam::Values& newValues() const { return new_values_; }
    [[nodiscard]] bool hasWork() const { return has_work_; }

    // Indices (into the iSAM2 factor graph) of previously-inserted smart
    // factors that we just superseded with a re-added version. Passed to
    // ISAM2::update()'s remove list so the old version is dropped.
    [[nodiscard]] const gtsam::FactorIndices& removeIndices() const {
        return remove_indices_;
    }
    // Per smart factor we pushed into new_factors_: a pair of
    // (position-in-new_factors, landmark_id). Used by the iSAM update
    // pass to back-fill MappingState::smart_factor_indices once iSAM
    // returns the new factor indices.
    [[nodiscard]] const std::vector<std::pair<size_t, LandmarkId>>&
    smartFactorPositions() const {
        return smart_factor_positions_;
    }

    void setStorage(AnyBag& storage) { storage_ = &storage; }

   private:
    MappingState& state_;
    Opts opts_;
    AnyBag* storage_ = nullptr;

    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    gtsam::FactorIndices remove_indices_;
    std::vector<std::pair<size_t, LandmarkId>> smart_factor_positions_;
    bool has_work_ = false;

    // Cached preintegration params (gravity, body_P_sensor, noise) used
    // for every CombinedImuFactor built by this pass. Lazily initialised
    // on the first keyframe so the gravity vector reflects the just-
    // computed startup gravity estimate in MappingState.
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params_;

    [[nodiscard]] std::optional<std::string> ensureCalibration();
    // Build `imu_params_` from MappingState's gravity + IMUSensorParams +
    // camera extrinsics. Called once on the first accepted keyframe.
    [[nodiscard]] std::optional<std::string> ensureImuParams();
};

}  // namespace wslam
