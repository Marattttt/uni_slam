#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <condition_variable>
#include <cstdint>
#include <expected>
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
    // Tuning knobs for the underlying gtsam::ISAM2 instance. Defaults mirror
    // the historically-hardcoded values; the mapping stage overrides them
    // from Isam2UpdatePass::Opts (themselves env-overridable for benchmark
    // sweeps). See the worker ctor for the rationale behind each default.
    struct Params {
        // Relinearise variables whose linear delta exceeds this. Larger =>
        // fewer variables relinearised per update => cheaper, looser. The
        // effective default is set by Isam2UpdatePass::Opts (0.1); see the
        // rationale there.
        double relinearize_threshold = 0.1;
        // Relinearise every Nth update. 1 = every update (most accurate,
        // most expensive).
        int relinearize_skip = 1;
        // false => QR factorisation (rank-revealing, numerically robust to
        // the IMU/vision information mismatch, but ~2-4x slower);
        // true => CHOLESKY (faster, but can hit indefinite pivots on long
        // VI runs — see ctor comment).
        bool use_cholesky = false;
        // Reuse freed factor slots so the smart-factor remove-and-readd
        // pattern doesn't grow the factor array without bound.
        bool find_unused_factor_slots = true;
    };

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

    explicit Isam2Worker(Params params);
    ~Isam2Worker();

    Isam2Worker(const Isam2Worker&) = delete;
    Isam2Worker& operator=(const Isam2Worker&) = delete;
    Isam2Worker(Isam2Worker&&) = delete;
    Isam2Worker& operator=(Isam2Worker&&) = delete;

    // Hand a unit of work to the worker thread. Returns a future the caller
    // must consume before the next submission, otherwise behaviour is
    // undefined (and an assert will fire in debug builds).
    [[nodiscard]] std::future<Result> submit(Work work);

    // Run one global batch Levenberg-Marquardt optimisation over the full
    // factor graph iSAM2 has accumulated, seeded by its current estimate.
    // Used at export time to recover the global optimum the (deliberately
    // cheaper) per-frame incremental updates may have left on the table.
    //
    // THREAD-SAFETY: this touches `isam_` directly from the *caller's*
    // thread. It is only safe to call when no work is in flight — i.e. after
    // the caller has drained the last submit() future and is not submitting
    // more. In that state the worker thread is parked in cv_.wait() and never
    // touches `isam_`, so there is no race. Calling it concurrently with an
    // in-flight submission is undefined.
    [[nodiscard]] std::expected<gtsam::Values, std::string> optimizeBatch(
        const gtsam::LevenbergMarquardtParams& lm_params);

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
