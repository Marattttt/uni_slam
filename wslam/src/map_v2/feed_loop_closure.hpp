#pragma once

#include <memory>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "loop_closure/loop_closure_worker.hpp"
#include "map_common.hpp"

namespace wslam::v2::mapping {

// Last pass of the mapping stage when loop closure is on (it only runs on
// accepted keyframes). Hands the loop closure thread what it needs and takes
// back its drift correction, never waiting for it:
// - the accepted keyframe's image and the landmarks it tracks
//   (SharedBindings::active_landmarks)
// - the pose the last drain settled, and the keyframes and landmarks the
//   drains archived since the previous feed
// - the newest drift correction, into SharedBindings::drift_correction
//
// Input  (from AnyBag): the frame under GetFrameName(0)
// Output (to   AnyBag): -
class FeedLoopClosurePass : public compute::Pass {
   public:
    FeedLoopClosurePass(AnyBag& storage, SharedBindings& shared,
                        std::shared_ptr<loop_closure::LoopClosureWorker> worker)
        : storage_(storage), shared_(shared), worker_(std::move(worker)) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

    // Sends the last drain's results. Call once after the mapping stage's
    // own flush, before finishing the worker
    void flush();

   private:
    AnyBag& storage_;
    SharedBindings& shared_;
    const std::shared_ptr<loop_closure::LoopClosureWorker> worker_;

    PoseId last_sent_drained_{0};
    bool gravity_sent_ = false;

    void pushMapUpdate();
    void pullCorrection();
};

}  // namespace wslam::v2::mapping
