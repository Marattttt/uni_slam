#pragma once

#include <cstdint>
#include <memory>

#include "common.hpp"
#include "mapping_state.hpp"
#include "pass.hpp"

namespace wslam {

// First pass of the mapping stage.
//
// Reads the per-frame TriangulationResult, decides whether the frame qualifies
// as a keyframe, allocates a fresh pose key for it, associates the new
// landmarks with either an existing track or a fresh LandmarkId, and writes a
// `MapDelta` describing the change for the downstream factor builder.
//
// Inputs  (AnyBag): TriangulationResult under TriangulationResultName.
// Outputs (AnyBag): MapDelta under MapDeltaName.
class KeyframeGatePass : public compute::Pass {
   public:
    struct Opts {
        // Reject frames whose triangulation produced fewer landmarks than
        // this. Below ~30 the optimisation becomes unstable on monocular.
        size_t min_landmarks = 30;
        // Minimum recovered relative rotation (radians) below which we
        // *also* require sufficient pixel parallax. Combined with
        // min_parallax_px the gate accepts EITHER significant rotation OR
        // significant translation, so pure-translation frames are still
        // admitted. ~0.02 rad ≈ 1.1°.
        double min_rotation_rad = 0.02;
        // Minimum median feature-pair pixel displacement (LOD-0 units)
        // below which — together with the rotation gate above — the frame
        // is considered too static to be useful.
        double min_parallax_px = 10.0;
        // Minimum number of *new* landmarks (i.e. tracks the previous
        // keyframe never saw) the frame must contribute to be accepted.
        // Stops admitting keyframes that are pure re-observations of
        // existing landmarks; those add factors but no map content.
        size_t min_new_landmarks = 5;
    };

    KeyframeGatePass(MappingState& state, std::shared_ptr<compute::GPU> gpu,
                     Opts opts);
    KeyframeGatePass(MappingState& state, std::shared_ptr<compute::GPU> gpu);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    // Hook used by the factory to bind to the AnyBag — passed in through the
    // pass and read straight from there.
    void setStorage(AnyBag& storage) { storage_ = &storage; }

   private:
    MappingState& state_;
    Opts opts_;
    AnyBag* storage_ = nullptr;
};

}  // namespace wslam
