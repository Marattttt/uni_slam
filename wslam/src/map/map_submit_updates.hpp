#pragma once

#include <gtsam/nonlinear/Values.h>

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "isam2_worker.hpp"
#include "map_common.hpp"
#include "pass.hpp"

namespace wslam::map {

// Final pass of the mapping stage.
//
// Consumes the per-keyframe `FactorBundle` that `BuildFactorsPass` published to
// the AnyBag, hands it to iSAM2, applies the optimised estimate back into the
// shared mapping state, and publishes a GTSAM-free `MapSnapshot` for downstream
// consumers (the GUI and the map exporter).
//
// Threading: works with an isam2 worker that has at most one work in flight and
// should be drained before this pass runs in a stage
//
// Inputs  (AnyBag): FactorBundle under ResourceIdentifier::FactorBundleName.
// Outputs (AnyBag): MapSnapshot under ResourceIdentifier::MapSnapshotName.
class SubmitUpdatesPass : public compute::Pass {
   public:
    struct Opts {
        // Threshold for when does a change in a variable's value lead to a
        // relinearization
        double relinearize_threshold = 0.01;
        // false => QR (rank-revealing, robust to the IMU/vision information
        // mismatch on long VI runs); true => Cholesky (faster where stable).
        bool use_cholesky = false;

        // Extra update() calles for further linearization of the isam2 graph
        uint32_t extra_updates = 2;

        // Publish a MapSnapshot every N accepted keyframes. Building a snapshot
        // re-triangulates every smart factor (O(map size)), so throttling
        // matters on long runs. 1 = every keyframe (the GUI wants this).
        // 0 = never during the loop — headless export runs, whose only consumer
        // is the post-loop snapshot, set this and rely on flush() publishing
        // the final one.
        uint32_t snapshot_every_n_keyframes = 0;
    };

    SubmitUpdatesPass(AnyBag& storage, MappingSharedBindings& shared, Opts opts)
        : storage_(storage), shared_(shared), opts_(opts) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

    /// Drain current work of isam2 worker
    ///
    /// Drain the previous keyframe's in-flight iSAM2 update: block until the
    /// worker finishes (a no-op when nothing is pending), apply the optimised
    /// estimate to the shared mapping state, and publish a MapSnapshot on the
    /// configured cadence. Public so `DrainUpdatesPass`, placed after the
    /// filter and before `BuildFactorsPass`, can run it — that way the builder
    /// reads up-to-date smart_factor_indices, optimised poses, and the
    /// propagated velocity/bias.
    [[nodiscard]] std::optional<std::string> drainPending();

    /// Wait for any in progress work and publish a snapshot
    ///
    /// Drain the final keyframe's optimisation and publish the final
    /// MapSnapshot. Call once after the pipeline loop exits and before the map
    /// exporter reads MapSnapshotName: `DrainUpdatesPass` never runs for the
    /// last keyframe (there is no next keyframe to trigger it), so this absorbs
    /// that pending result and guarantees the exported snapshot reflects the
    /// last keyframe even when `snapshot_every_n_keyframes` throttled the
    /// in-loop cadence.
    [[nodiscard]] std::optional<std::string> flush();

   private:
    AnyBag& storage_;
    MappingSharedBindings& shared_;
    const Opts opts_;

    std::unique_ptr<Isam2Worker> worker_;

    // Latest optimised estimate
    gtsam::Values latest_values_;

    // Total factors iSAM2 reported after the most recent update; stamped into
    // published snapshots.
    std::size_t last_factor_count_ = 0;

    // Accepted keyframes since the last published snapshot, compared against
    // Opts::snapshot_every_n_keyframes.
    uint32_t keyframes_since_snapshot_ = 0;

    // In-flight iSAM2 update submitted by execute() and awaited by
    // drainPending(). Invalid (default-constructed / already consumed) when
    // nothing is pending — the state before the first keyframe and after each
    // drain.
    std::future<Isam2Worker::Result> pending_;
    // Captured at submit time alongside `pending_`: (position-in-graph,
    // landmark) pairs so drainPending() can map iSAM2's returned FactorIndex
    // assignments back into per-landmark smart_factor_indices, and the pose id
    // whose velocity/bias to propagate. shared_.last_pose cannot be read at
    // drain time — the filter has already advanced it to the next keyframe.
    std::vector<std::pair<size_t, LandmarkId>> pending_smart_factor_positions_;
    PoseId pending_pose_id_{0};

    /// Rebuild the MapSnapshot from shared_ + latest_values_ and publish it
    /// under MapSnapshotName. Shared by the throttled in-loop path and flush().
    void publishSnapshot();
};

// Sibling pass that drains the previous keyframe's in-flight iSAM2 result by
// delegating to SubmitUpdatesPass. The stage places it *after*
// `FilterKeyframePass` (so it only runs on accepted keyframes, letting the
// inter-keyframe gap overlap the async solve) but *before* `BuildFactorsPass`
// (so the builder reads up-to-date smart-factor indices, optimised poses, and
// the propagated velocity/bias). See SubmitUpdatesPass::drainPending().
class DrainUpdatesPass : public compute::Pass {
   public:
    explicit DrainUpdatesPass(SubmitUpdatesPass& target) : target_(target) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

   private:
    SubmitUpdatesPass& target_;
};

}  // namespace wslam::map
