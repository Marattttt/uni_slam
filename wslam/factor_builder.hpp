#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <memory>

#include "anybag.hpp"
#include "mapping_state.hpp"
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

        // Prior noise on the very first landmark (tight — scale lock for
        // monocular). Without this the global scale is unobservable.
        double prior_landmark_sigma = 1e-2;

        // Prior noise on every other landmark introduced *on the first
        // keyframe*. Those landmarks only have one projection observation
        // (the prev camera pose lives outside the graph), so we anchor
        // them with their triangulated positions. The optimiser is still
        // free to refine them once they get observations from later
        // keyframes — but the linear system stays well posed in the
        // meantime.
        double first_keyframe_landmark_sigma = 0.5;

        // Soft position prior attached to every *other* newly inserted
        // landmark (those introduced on a later keyframe). Two projection
        // observations are usually enough to triangulate, but in
        // low-parallax cases the landmark's marginal Hessian becomes
        // near-singular and iSAM2 throws an IndeterminantLinearSystem
        // exception. A modest anchor on the triangulated position keeps
        // the linear system well conditioned without significantly
        // biasing the optimiser.
        double landmark_anchor_sigma = 2.0;

        // Between-pose factor noise (rotation in radians, translation in
        // monocular scale units — typically dimensionless).
        double between_rotation_sigma_rad = 0.05;
        double between_translation_sigma = 0.10;

        // Pixel noise (sigma) for projection factors at LOD-0.
        double projection_pixel_sigma = 1.5;

        // Use Huber robust kernel on projection factors? Strongly recommended
        // — outlier observations slipping past RANSAC otherwise pull the
        // optimisation off the global minimum.
        bool use_robust_projection = true;
        double huber_threshold_px = 1.345;  // standard 95%-eff value
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

    void setStorage(AnyBag& storage) { storage_ = &storage; }

   private:
    MappingState& state_;
    Opts opts_;
    AnyBag* storage_ = nullptr;

    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    bool has_work_ = false;

    [[nodiscard]] std::optional<std::string> ensureCalibration();
};

}  // namespace wslam
