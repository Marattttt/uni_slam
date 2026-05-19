#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace wslam {

// Background worker that owns a `gtsam::ISAM2` instance and runs the
// (potentially expensive) `update()` call off the main pipeline thread.
//
// Concurrency model: the worker accepts at most one in-flight job at a
// time. Backpressure is the caller's responsibility — wait on the future
// returned by the previous `submit()` before calling `submit()` again. The
// iSAM update pass enforces this by always draining the previous frame's
// future before submitting the current frame's work, which keeps the main
// thread at most one frame ahead of the optimiser.
class Isam2Worker {
   public:
    struct Work {
        gtsam::NonlinearFactorGraph new_factors;
        gtsam::Values new_values;
        // Indices of previously-inserted factors to remove on this
        // update — used by the smart-factor remove-and-readd pattern in
        // factor_builder. Empty when no factors need replacing.
        gtsam::FactorIndices remove_factor_indices;
        uint32_t extra_updates = 0;
        // Logging only — identifies which frame the submission came from.
        uint64_t frame_id = 0;
    };

    struct Result {
        // Optimised estimate after the update completes. Empty when `error`
        // is set.
        gtsam::Values latest_values;
        // Total factors in the iSAM2 graph after this update.
        std::size_t factor_count = 0;
        // For each entry in Work::new_factors (positionally), the
        // FactorIndex iSAM2 assigned it. The factor builder uses this to
        // record per-landmark factor indices for the next remove-and-
        // readd cycle. Same size as Work::new_factors.
        std::vector<gtsam::FactorIndex> new_factor_indices;
        // If non-empty, iSAM2 threw and the update was not applied.
        std::optional<std::string> error;
    };

    Isam2Worker();
    ~Isam2Worker();

    Isam2Worker(const Isam2Worker&) = delete;
    Isam2Worker& operator=(const Isam2Worker&) = delete;
    Isam2Worker(Isam2Worker&&) = delete;
    Isam2Worker& operator=(Isam2Worker&&) = delete;

    // Hand a unit of work to the worker thread. Returns a future the caller
    // must consume before the next submission, otherwise behaviour is
    // undefined (and an assert will fire in debug builds).
    [[nodiscard]] std::future<Result> submit(Work work);

   private:
    void run();

    std::unique_ptr<gtsam::ISAM2> isam_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::optional<std::pair<Work, std::promise<Result>>> pending_;
    bool stop_ = false;
};

}  // namespace wslam
