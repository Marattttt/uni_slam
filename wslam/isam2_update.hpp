#pragma once

#include <memory>

#include "anybag.hpp"
#include "factor_builder.hpp"
#include "mapping_state.hpp"
#include "pass.hpp"

namespace wslam {

// Third pass of the mapping stage.
//
// Reads the per-frame factor + value delta accumulated by the factor builder,
// pushes it through iSAM2.update(), extracts the current estimate, and
// publishes a `MapSnapshot` to AnyBag for downstream consumers (GUI,
// telemetry). The iSAM2 instance and the latest_values cache on
// MappingState are mutated in place.
//
// Inputs  (in-process): FactorBuilderPass::newFactors / newValues
// Outputs (AnyBag):     MapSnapshot under ResourceIdentifier::MapSnapshotName
class Isam2UpdatePass : public compute::Pass {
   public:
    struct Opts {
        // Extra optimisation steps per update. iSAM2's docs recommend 0–2.
        // More is wasted work if relinearisation converges quickly; on a
        // monocular front-end an extra step typically helps when many new
        // observations were just added.
        uint32_t extra_updates = 1;
    };

    Isam2UpdatePass(MappingState& state, const FactorBuilderPass& builder,
                    std::shared_ptr<compute::GPU> gpu, Opts opts);
    Isam2UpdatePass(MappingState& state, const FactorBuilderPass& builder,
                    std::shared_ptr<compute::GPU> gpu);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    void setStorage(AnyBag& storage) { storage_ = &storage; }

   private:
    MappingState& state_;
    const FactorBuilderPass& builder_;
    Opts opts_;
    AnyBag* storage_ = nullptr;
};

}  // namespace wslam
