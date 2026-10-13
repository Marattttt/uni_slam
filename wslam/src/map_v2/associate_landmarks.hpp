#pragma once

#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "map_common.hpp"

namespace wslam::v2::mapping {

// Turns the accepted keyframe's matches into landmark observations.
//
// Runs after the previous smoother update was drained, so it sees which
// landmarks the smoother marginalised. A match continues a landmark when its
// reference keypoint observed one in the last keyframe (looked up by keypoint
// index, confirmed by pixel, see LookupTrack) and that landmark's smart factor
// is still in the window; it then observes it from the new pose only. Any
// other match starts a new landmark observed from both keyframes (only the new
// one on the first keyframe). A landmark whose factor left the window is not
// continued: its information, including the previous keyframe's observation,
// is in the marginal, so its successor starts from the new keyframe alone, and
// matching it to the old landmark is left to re-observation handling outside
// the local smoother.
//
// Rebuilds SharedBindings::active_landmarks for the new keyframe.
//
// Input  (from AnyBag): KeyframeCandidate under KeyframeCandidateName (taken)
// Output (to   AnyBag): MapChanges under MapDeltaName
class AssociateLandmarksPass : public compute::Pass {
   public:
    struct Opts {
        // See FilterKeyframeConfig::association_tolerance_px
        double association_tolerance_px = 0.5;
    };

    AssociateLandmarksPass(AnyBag& storage, SharedBindings& shared, Opts opts)
        : storage_(storage), shared_(shared), opts_(opts) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

   private:
    AnyBag& storage_;
    SharedBindings& shared_;
    const Opts opts_;
};

}  // namespace wslam::v2::mapping
