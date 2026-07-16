#pragma once

#include <cstdint>
#include <memory>

#include "common.hpp"
#include "mapping_state.hpp"
#include "compute/pass.hpp"

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
        // admitted. ~0.05 rad ≈ 2.9° — V101 yaws at up to ~0.5 rad/s, so
        // anything much smaller accepts a keyframe every single frame
        // during rotation segments.
        double min_rotation_rad = 0.05;
        // Minimum median feature-pair pixel displacement (LOD-0 units)
        // below which — together with the rotation gate above — the frame
        // is considered too static to be useful. Measured against the
        // previous KEYFRAME (the reference set), so parallax accumulates
        // across frames; 20 px spaces keyframes ~3-5 frames apart on
        // V101, giving each graph edge a usable baseline instead of
        // accepting nearly every frame.
        double min_parallax_px = 20.0;
        // Minimum number of *new* landmarks (i.e. tracks the previous
        // keyframe never saw) the frame must contribute to be accepted.
        // Stops admitting keyframes that are pure re-observations of
        // existing landmarks; those add factors but no map content.
        size_t min_new_landmarks = 5;
        // Per-landmark minimum pixel parallax (LOD-0). NEW landmarks whose
        // feat_prev ↔ feat_curr displacement is below this are dropped
        // before entering the graph — they're depth-ambiguous (focus-of-
        // expansion or distant points) and tend to give iSAM2 singular
        // marginal Hessians even with a soft anchor prior. Re-observations
        // are not filtered: a previously-triangulated landmark constrains
        // the new pose regardless of inter-frame motion.
        double min_landmark_parallax_px = 10.0;

        // Median pixel parallax (vs the previous *frame* — pre-origin
        // matching is frame-to-frame) required to accept the FIRST
        // keyframe. Keeps the gauge anchor off the stationary lead-in:
        // the first keyframe lands at motion onset, so the stationary
        // window behind it gives a clean gravity/gyro-bias estimate, the
        // zero-velocity prior is still approximately true, and the first
        // IMU preintegration interval stays short. Much smaller than
        // min_parallax_px because one inter-frame step at motion onset
        // moves features only a few pixels.
        double bootstrap_parallax_px = 3.0;

        // Gravity / gyro-bias initialisation window. The gate picks the
        // *quietest* window of this many samples from the accumulated IMU
        // buffer (minimum stationarity score = std(|accel|) + mean gyro
        // deviation from the window mean) instead of blindly averaging
        // the whole buffer — the buffer ends at the first keyframe, i.e.
        // at motion onset, so its tail is contaminated by the takeoff.
        // Absolute thresholds don't work here: a grounded MAV with
        // spinning props vibrates (V101: std|accel| ≈ 0.2–0.3 m/s²
        // while parked) and EuRoC's gyro bias alone is ~0.08 rad/s.
        size_t gravity_window_samples = 200;  // ~1 s at 200 Hz
        // Score above which even the quietest window is reported as
        // non-stationary (V101 parked ≈ 0.25–0.55; in flight ≥ 1.4).
        double stationary_max_score = 0.8;
    };

    KeyframeGatePass(std::shared_ptr<MappingState> state, Opts opts);
    KeyframeGatePass(std::shared_ptr<MappingState> state);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

    // Hook used by the factory to bind to the AnyBag — passed in through the
    // pass and read straight from there.
    void setStorage(AnyBag& storage) { storage_ = &storage; }

   private:
    std::shared_ptr<MappingState> state_;
    Opts opts_;
    AnyBag* storage_ = nullptr;
};

}  // namespace wslam
