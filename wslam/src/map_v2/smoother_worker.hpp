#pragma once

#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

#include "job_worker.hpp"
#include "map_common.hpp"

namespace wslam::v2::mapping {

// Batch fixed-lag smoother whose factor removal frees the slots it empties.
//
// The base update(..., factorsToRemove) only resets the removed slots: it
// neither hands them back for reuse nor drops them from its key -> factor
// index, so the smart-factor remove-and-re-add pattern would grow the graph by
// a null slot per re-observation, forever. This update removes through the
// protected removeFactors() first, which does both, then inserts the new
// factors (into the freed slots first) and runs the base update
class LocalSmoother : public gtsam::BatchFixedLagSmoother {
   public:
    using gtsam::BatchFixedLagSmoother::BatchFixedLagSmoother;
    // Base overloads stay visible; remove factors through the one below
    using gtsam::BatchFixedLagSmoother::update;

    Result update(const gtsam::NonlinearFactorGraph& new_factors,
                  const gtsam::Values& new_values,
                  const KeyTimestampMap& timestamps,
                  const std::set<size_t>& remove_slots);
};

// Owns a LocalSmoother and runs its updates on a dedicated thread, one job in
// flight (base::JobWorker). The smoother is only ever touched by that thread;
// results are copied out.
//
// Sharing rule for the factors in a job: once submitted, a factor is owned by
// the smoother and must not be mutated by anyone else (re-observations clone
// it). Reading such a factor (e.g. SmartFactor::point(), which updates its
// internal cache) is only safe while no job is running, i.e. after the
// previous job's future was consumed. Other threads (e.g. a future
// re-observation worker) must work from copies and the archive instead
class SmootherWorker {
   public:
    struct Params {
        // Keyframes kept in the window; older ones are marginalised. Keys are
        // stamped with SmootherTime(), so the smoother lag is this minus one
        uint32_t window_keyframes = 10;
        // Levenberg-Marquardt iterations per update
        uint32_t max_iterations = 10;
        // Keep the linearisation point of variables touched by marginal
        // factors fixed, as the marginals were computed there
        bool enforce_consistency = true;
    };

    struct Work {
        FactorBundle bundle;
        // Logging only
        uint64_t keyframe_id = 0;
    };

    struct Result {
        // Estimate of every variable still in the window. Empty on error
        gtsam::Values latest_values;
        // Slot of every factor in the smoother after the update, by identity.
        // A submitted factor missing here was marginalised
        std::unordered_map<const gtsam::NonlinearFactor*, size_t> live_slots;
        size_t factor_count = 0;
        size_t iterations = 0;
        double graph_error = 0.0;
        // Set when the update threw; the smoother is then unusable
        std::optional<std::string> error;
    };

    explicit SmootherWorker(Params params);

    // Must not be called while the previous job's future is unconsumed
    [[nodiscard]] std::future<Result> submit(Work work);

    [[nodiscard]] const Params& params() const { return params_; }

   private:
    const Params params_;
    // Declared before worker_ so it outlives the worker thread
    std::unique_ptr<LocalSmoother> smoother_;
    base::JobWorker<Work, Result> worker_;

    [[nodiscard]] static Result Run(LocalSmoother& smoother, Work&& work);
};

}  // namespace wslam::v2::mapping
