#pragma once

#include <spdlog/spdlog.h>

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "models.hpp"

#include "anybag.hpp"
#include "factor_builder.hpp"
#include "isam2_worker.hpp"
#include "mapping_state.hpp"
#include "compute/pass.hpp"

namespace wslam {

// Third pass of the mapping stage.
//
// Reads the per-frame factor + value delta accumulated by the factor builder
// and hands it to an `Isam2Worker` running on a dedicated thread. The pass
// is asynchronous: each execute() drains the *previous* frame's pending
// result (blocking only if iSAM hasn't caught up), applies it to
// `MappingState::latest_values`, publishes a `MapSnapshot`, and then submits
// the current frame's work. This keeps the main pipeline thread at most one
// frame ahead of the optimiser.
//
// Inputs  (in-process): FactorBuilderPass::newFactors / newValues
// Outputs (AnyBag):     MapSnapshot under ResourceIdentifier::MapSnapshotName
class Isam2UpdatePass : public compute::Pass {
   public:
    struct Opts {
        // Extra optimisation steps per update. iSAM2's docs recommend 0–2.
        // Default 0 keeps the per-frame back-end cost minimal; the worker
        // already runs the main update() and that's typically enough on
        // its own. Bump to 1–2 only if profiling shows large post-relin
        // residuals.
        uint32_t extra_updates = 0;

        // Publish a fresh MapSnapshot every N successful drains. Building
        // the snapshot re-triangulates every smart factor ever created
        // (O(map size), growing without bound) on the main thread.
        // 0 => never publish during the loop: headless runs whose only
        // snapshot consumer is the post-loop map export use this. The GUI
        // reads the snapshot every frame and wants 1. flush() always
        // publishes regardless, so the exported map never lags.
        uint32_t snapshot_every_n_drains = 1;

        // --- Live-loop iSAM2 tuning (forwarded to Isam2Worker::Params) ---
        // Larger threshold / skip => fewer, cheaper relinearisations =>
        // a looser ("partial") live estimate that the final batch
        // optimisation below recovers. See Isam2Worker::Params.
        //
        // relinearize_threshold defaults to 0.1 (raised from the textbook
        // 0.01): at 0.01 iSAM2 relinearises so aggressively that the cascade
        // re-eliminates the whole Bayes tree every keyframe (~90k factors by
        // frame 700 on V101), which is what made the drain pass scale with
        // map size. 0.1 shrinks that cascade and cut the mapping stage ~85%
        // at 700 frames with RPE unchanged and APE within run-to-run noise
        // (benchmarked; QR + skip=1 kept — Cholesky crashes on long VI runs
        // and skip>1 degraded accuracy without extra speedup over threshold).
        double relinearize_threshold = 0.1;
        int relinearize_skip = 1;
        bool use_cholesky = false;  // false => QR

        // --- Final batch optimisation at flush/export ---
        // Run one global Levenberg-Marquardt over iSAM2's retained graph at
        // flush(), seeded by the incremental estimate, before publishing the
        // exported snapshot.
        //
        // DISABLED by default: it is actively harmful on long monocular-VI
        // sequences. SmartProjectionPoseFactors are scale-free, and they
        // outnumber the scale-bearing CombinedImuFactors ~350:1 (one IMU
        // factor per keyframe edge vs one smart factor per landmark — ~87k
        // landmarks vs 243 IMU edges on full V101). A free global solve lets
        // the visual factors collectively rescale the map, collapsing metric
        // scale (measured Umeyama scale 1.13 -> 0.001, APE 0.33 m -> 1.42 m
        // on full V101). The sequential incremental estimate resists this
        // because each keyframe is anchored step-by-step, so it already holds
        // scale and accuracy without a batch pass. Left as an opt-in
        // (WSLAM_FINAL_BATCH=1) for setups that anchor scale independently.
        // On failure the live estimate is exported instead (map never lost).
        bool final_batch_optimize = false;
        int final_batch_max_iterations = 100;
    };

    Isam2UpdatePass(std::shared_ptr<MappingState> state,
                    const FactorBuilderPass& builder, Opts opts);
    Isam2UpdatePass(std::shared_ptr<MappingState> state,
                    const FactorBuilderPass& builder);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    void setStorage(AnyBag& storage) { storage_ = &storage; }

    // Drain any in-flight iSAM update and publish its result to
    // MappingState + MapSnapshot. Public so a sibling Pass placed at the
    // *start* of the mapping stage can run this before KeyframeGate /
    // FactorBuilder run — that way the factor builder reads up-to-date
    // smart_factor_indices and the gate chains off optimised poses.
    [[nodiscard]] std::optional<std::string> drainPending();

    // Blocks until any in-flight iSAM update completes, applies the result
    // to MappingState, and publishes the final snapshot. Call once after
    // the main pipeline loop exits and before consumers (e.g. ExportMap)
    // read MapSnapshotName, otherwise the final keyframe's optimisation
    // will be lost.
    [[nodiscard]] std::optional<std::string> flush();

   private:
    std::shared_ptr<MappingState> state_;
    const FactorBuilderPass& builder_;
    Opts opts_;
    AnyBag* storage_ = nullptr;

    // Rebuild the MapSnapshot from MappingState and publish it under
    // MapSnapshotName. Factored out so drainPending (throttled) and
    // flush (always) share one implementation.
    void publishSnapshot();

    std::unique_ptr<Isam2Worker> worker_;
    std::future<Isam2Worker::Result> pending_;
    // Drains applied since the last published snapshot, compared against
    // Opts::snapshot_every_n_drains.
    uint32_t drains_since_snapshot_ = 0;
    // Factor count reported by the most recent worker result; stamped
    // into snapshots built outside drainPending (e.g. at flush time).
    std::size_t last_factor_count_ = 0;
    // Snapshot of FactorBuilderPass::smartFactorPositions() at the moment
    // we submitted the previous frame's work. Held alongside `pending_`
    // so drainPending can map iSAM's returned FactorIndex assignments
    // back into per-landmark indices in MappingState.
    std::vector<std::pair<size_t, LandmarkId>> pending_smart_factor_positions_;
    uint64_t frame_counter_ = 0;
};

// Sibling pass that drains any in-flight iSAM2 result by delegating to
// an existing Isam2UpdatePass. Placed at the start of the mapping stage
// so the keyframe gate / factor builder see up-to-date estimates and
// smart_factor_indices before they run.
class Isam2DrainPass : public compute::Pass {
   public:
    Isam2DrainPass(Isam2UpdatePass& target) : target_(target) {}

    [[nodiscard]] std::optional<std::string> initialize() override {
        return std::nullopt;
    }
    [[nodiscard]] std::optional<std::string> execute() override {
        spdlog::info("[ISAM2 Drain pass] Executing");
        return target_.drainPending();
    }
    [[nodiscard]] std::string getId() const override {
        return "[ISAM2 Drain pass]";
    }

   private:
    Isam2UpdatePass& target_;
};

}  // namespace wslam
