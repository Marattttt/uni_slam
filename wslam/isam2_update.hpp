#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <optional>

#include "anybag.hpp"
#include "factor_builder.hpp"
#include "isam2_worker.hpp"
#include "mapping_state.hpp"
#include "pass.hpp"

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
    };

    Isam2UpdatePass(MappingState& state, const FactorBuilderPass& builder,
                    std::shared_ptr<compute::GPU> gpu, Opts opts);
    Isam2UpdatePass(MappingState& state, const FactorBuilderPass& builder,
                    std::shared_ptr<compute::GPU> gpu);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    void setStorage(AnyBag& storage) { storage_ = &storage; }

    // Blocks until any in-flight iSAM update completes, applies the result
    // to MappingState, and publishes the final snapshot. Call once after
    // the main pipeline loop exits and before consumers (e.g. ExportMap)
    // read MapSnapshotName, otherwise the final keyframe's optimisation
    // will be lost.
    [[nodiscard]] std::optional<std::string> flush();

   private:
    MappingState& state_;
    const FactorBuilderPass& builder_;
    Opts opts_;
    AnyBag* storage_ = nullptr;

    std::unique_ptr<Isam2Worker> worker_;
    std::future<Isam2Worker::Result> pending_;
    uint64_t frame_counter_ = 0;

    // Block on the previous frame's future (if any), apply its result to
    // MappingState, and publish the resulting MapSnapshot. No-op when no
    // submission is in flight.
    [[nodiscard]] std::optional<std::string> drainPending();
};

}  // namespace wslam
