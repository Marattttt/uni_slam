#pragma once

#include <gtsam/nonlinear/Values.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "map_common.hpp"
#include "smoother_worker.hpp"

namespace wslam::v2::mapping {

// Last pass of the mapping stage: hands the keyframe's FactorBundle to the
// smoother worker without waiting for it. The result is applied by
// drainPending(), run by DrainSmootherPass on the next accepted keyframe (or
// by flush() after the loop), so the frames in between overlap the solve.
//
// Input  (from AnyBag): FactorBundle under FactorBundleName (taken)
// Output (to   AnyBag): MapSnapshot under MapSnapshotName, on cadence
class SubmitSmootherPass : public compute::Pass {
   public:
    struct Opts {
        SmootherWorker::Params smoother;

        // Publish a MapSnapshot every N drained keyframes; 0 = only in
        // flush(). Building one triangulates every in-window smart factor
        uint32_t snapshot_every_n_keyframes = 0;
    };

    SubmitSmootherPass(AnyBag& storage, SharedBindings& shared, Opts opts)
        : storage_(storage), shared_(shared), opts_(opts) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

    /// Wait for the in-flight update (a no-op when there is none) and apply
    /// it to the shared state:
    /// - keyframes that left the window are archived with the estimate they
    ///   had before it (the previous drain's)
    /// - every landmark whose smart factor is no longer in the smoother was
    ///   marginalised: its last valid triangulation is archived and it leaves
    ///   the window bookkeeping; the others get their current slot
    /// - the drained keyframe's velocity and bias are propagated
    /// - a snapshot is published on cadence
    [[nodiscard]] std::optional<std::string> drainPending();

    /// Drain the last update and publish the final snapshot. Call once after
    /// the pipeline loop, before the map is exported
    [[nodiscard]] std::optional<std::string> flush();

    /// Estimate of the window after the last drained update
    [[nodiscard]] const gtsam::Values& latestValues() const {
        return latest_values_;
    }

   private:
    AnyBag& storage_;
    SharedBindings& shared_;
    const Opts opts_;

    std::unique_ptr<SmootherWorker> worker_;

    // Estimate of the window after the last drained update
    gtsam::Values latest_values_;
    size_t last_factor_count_ = 0;
    uint32_t keyframes_since_snapshot_ = 0;

    // In flight: submitted by execute(), consumed by drainPending()
    std::future<SmootherWorker::Result> pending_;
    PoseId pending_pose_id_{0};
    // Folds the drained keyframe's velocity error against its IMU prediction
    // into SharedBindings::velocity_error_variance
    void trackVelocityError();

    void archiveMarginalisedPoses(const gtsam::Values& updated);
    void reconcileSmartFactors(const SmootherWorker::Result& result);
    void publishSnapshot();
};

// Drains the previous keyframe's smoother update by delegating to
// SubmitSmootherPass::drainPending(). Placed after the keyframe filter (so it
// only blocks on accepted keyframes) and before landmark association and
// factor building (so both see the window as the smoother left it)
class DrainSmootherPass : public compute::Pass {
   public:
    explicit DrainSmootherPass(SubmitSmootherPass& target) : target_(target) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

   private:
    SubmitSmootherPass& target_;
};

}  // namespace wslam::v2::mapping
