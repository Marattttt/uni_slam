#pragma once

#include <gtsam/nonlinear/Values.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "isam2_worker.hpp"
#include "map_common.hpp"
#include "pass.hpp"

namespace wslam::map {

// Final pass of the mapping stage.
//
// Consumes the per-keyframe `FactorBundle` that `BuildFactorsPass` published to
// the AnyBag, feeds it to iSAM2, applies the optimised estimate back into the
// shared mapping state, and publishes a GTSAM-free `MapSnapshot` for downstream
// consumers (the GUI and the map exporter).
//
// Threading: the iSAM2 solve itself runs on a dedicated `Isam2Worker` thread,
// but this pass drives it *synchronously* — it submits the bundle and blocks on
// the result inside a single execute(). This is what lets the whole update
// collapse into one pass (rather than the split drain-then-submit the older
// asynchronous design needed): because `FilterKeyframePass` stops the stage on
// rejected frames, this pass only runs on accepted keyframes, and by waiting
// for each result before returning we guarantee the next keyframe's
// `BuildFactorsPass` sees fully up-to-date smart-factor indices and
// bundle-adjusted poses. The worker is retained (rather than calling iSAM2
// inline) so its exception formatting / timing logs are reused and so the
// pass can be split back into an asynchronous drain+submit pair later without
// touching the solver code.
//
// Inputs  (AnyBag): FactorBundle under ResourceIdentifier::FactorBundleName.
// Outputs (AnyBag): MapSnapshot under ResourceIdentifier::MapSnapshotName.
class SubmitUpdatesPass : public compute::Pass {
   public:
    struct Opts {
        // --- iSAM2 solver tuning (forwarded to Isam2Worker::Params) ---
        // Relinearise variables whose linear delta exceeds this. Kept at 0.1
        // (raised from the textbook 0.01): at 0.01 iSAM2 re-eliminates almost
        // the whole Bayes tree every keyframe, which is what made the update
        // cost scale with map size; 0.1 shrinks that cascade ~85% with RPE
        // unchanged (see project_mapping_backend_perf / HOWTO_REEVALUATE).
        double relinearize_threshold = 0.1;
        // Relinearise every Nth update. 1 = every update (most accurate).
        int relinearize_skip = 1;
        // false => QR (rank-revealing, robust to the IMU/vision information
        // mismatch on long VI runs); true => Cholesky (faster where stable).
        bool use_cholesky = false;
        // Extra iSAM2 iterations after the main update. 0 keeps the per-frame
        // back-end cost minimal; bump only if profiling shows large residuals.
        uint32_t extra_updates = 0;

        // Publish a MapSnapshot every N accepted keyframes. Building a snapshot
        // re-triangulates every smart factor (O(map size)), so throttling
        // matters on long runs. 1 = every keyframe (the GUI wants this).
        // 0 = never during the loop — headless export runs, whose only consumer
        // is the post-loop snapshot, set this and rely on flush() publishing
        // the final one.
        uint32_t snapshot_every_n_keyframes = 1;
    };

    SubmitUpdatesPass(AnyBag& storage, MappingSharedBindings& shared, Opts opts)
        : storage_(storage), shared_(shared), opts_(opts) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

    /// Publish the final MapSnapshot. Call once after the pipeline loop exits
    /// and before the map exporter reads MapSnapshotName. In this synchronous
    /// design there is no in-flight optimisation left to drain, so this only
    /// guarantees the exported snapshot reflects the last keyframe even when
    /// `snapshot_every_n_keyframes` throttled the in-loop cadence.
    [[nodiscard]] std::optional<std::string> flush();

   private:
    AnyBag& storage_;
    MappingSharedBindings& shared_;
    const Opts opts_;

    // Owns the gtsam::ISAM2 instance. Created in initialize() from Opts.
    std::unique_ptr<Isam2Worker> worker_;

    // Latest optimised estimate, cached on the main thread after each update.
    // Only this pass needs it (for snapshot building and for mirroring poses
    // into shared_.predicted_values), so it is not part of the shared state.
    gtsam::Values latest_values_;

    // Total factors iSAM2 reported after the most recent update; stamped into
    // published snapshots.
    std::size_t last_factor_count_ = 0;

    // Accepted keyframes since the last published snapshot, compared against
    // Opts::snapshot_every_n_keyframes.
    uint32_t keyframes_since_snapshot_ = 0;

    /// Rebuild the MapSnapshot from shared_ + latest_values_ and publish it
    /// under MapSnapshotName. Shared by the throttled in-loop path and flush().
    void publishSnapshot();
};

}  // namespace wslam::map
