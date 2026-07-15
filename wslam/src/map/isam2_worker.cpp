#include "isam2_worker.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <chrono>
#include <format>
#include <typeinfo>
#include <utility>

#include "assert.hpp"

using namespace wslam;

#define LOG_ID "[ISAM2 worker]"

Isam2Worker::Isam2Worker(Params worker_params) {
    gtsam::ISAM2Params params;
    params.optimizationParams = gtsam::ISAM2GaussNewtonParams();
    params.relinearizeThreshold = worker_params.relinearize_threshold;
    params.factorization = gtsam::ISAM2Params::CHOLESKY;

    // The smart-factor remove-and-readd pattern otherwise leaves a dead
    // slot in the factor graph per re-observed landmark per keyframe,
    // growing the factor array (and VariableIndex bookkeeping) without
    // bound. Slot reuse keeps ISAM2Result::newFactorsIndices positional,
    // so the per-landmark index backfill in the update pass still holds.
    params.findUnusedFactorSlots = worker_params.find_unused_factor_slots;
    isam_ = std::make_unique<gtsam::ISAM2>(params);

    thread_ = std::thread([this] { run(); });
    spdlog::info(LOG_ID
                 " Worker thread started (factorization={}, relin_thresh={})",
                 worker_params.use_cholesky ? "CHOLESKY" : "QR",
                 worker_params.relinearize_threshold);
}

Isam2Worker::~Isam2Worker() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    spdlog::info(LOG_ID " Worker thread joined");
}

std::future<Isam2Worker::Result> Isam2Worker::submit(Work work) {
    std::promise<Result> p;
    auto fut = p.get_future();
    {
        std::lock_guard<std::mutex> lk(mu_);
        WSLAM_ASSERT(!pending_.has_value(),
                     "Isam2Worker: previous submission still pending");
        pending_.emplace(std::move(work), std::move(p));
    }
    cv_.notify_one();
    return fut;
}

void Isam2Worker::run() {
    while (true) {
        std::optional<std::pair<Work, std::promise<Result>>> item;
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_ || pending_.has_value(); });
            if (stop_ && !pending_.has_value()) {
                return;
            }
            item = std::move(pending_);
            pending_.reset();
        }

        Result result;
        try {
            spdlog::debug(LOG_ID
                          " frame={}: submitting factors={}, new_values={}, "
                          "remove={}",
                          item->first.frame_id, item->first.new_factors.size(),
                          item->first.new_values.size(),
                          item->first.remove_factor_indices.size());
            using Clock = std::chrono::steady_clock;
            const auto ms = [](Clock::duration d) {
                return std::chrono::duration<double, std::milli>(d).count();
            };
            const auto t_update_begin = Clock::now();
            const auto r
                = isam_->update(item->first.new_factors, item->first.new_values,
                                item->first.remove_factor_indices);
            const auto t_update_end = Clock::now();
            for (uint32_t i = 0; i < item->first.extra_updates; ++i) {
                isam_->update();
            }
            const auto t_extra_end = Clock::now();
            result.latest_values = isam_->calculateEstimate();
            const auto t_estimate_end = Clock::now();
            result.factor_count = isam_->getFactorsUnsafe().size();
            // ISAM2Result::newFactorsIndices is a vector mapping the
            // position of each newly inserted factor (in order) to its
            // FactorIndex in the underlying NonlinearFactorGraph. Copy
            // it out so the main thread can update its per-landmark
            // index map.
            result.new_factor_indices.assign(r.newFactorsIndices.begin(),
                                             r.newFactorsIndices.end());
            // update_ms is the dominant, map-size-scaling cost; estimate_ms
            // is the full Values extraction (only O(#variables) — small);
            // total_factors / total_values are the map-size proxies to
            // correlate against. See benchmarks/HOWTO_REEVALUATE for usage.
            spdlog::info(
                LOG_ID
                " frame={} OK: relin={}, factors_recalc={}, total_factors={}, "
                "total_values={}, update_ms={:.1f}, extra_ms={:.1f}, "
                "estimate_ms={:.1f}",
                item->first.frame_id, r.variablesRelinearized,
                r.factorsRecalculated, result.factor_count,
                result.latest_values.size(), ms(t_update_end - t_update_begin),
                ms(t_extra_end - t_update_end),
                ms(t_estimate_end - t_extra_end));
        } catch (const gtsam::IndeterminantLinearSystemException& e) {
            result.error = std::format(
                "iSAM2 update: indeterminate linear system at key '{}' (chr "
                "'{}' index {}). Likely cause: a variable has no constraints "
                "yet. what(): {}",
                gtsam::DefaultKeyFormatter(e.nearbyVariable()),
                static_cast<char>(gtsam::Symbol(e.nearbyVariable()).chr()),
                gtsam::Symbol(e.nearbyVariable()).index(), e.what());
        } catch (const std::exception& e) {
            result.error = std::format("iSAM2 update threw [{}]: {}",
                                       typeid(e).name(), e.what());
        }
        item->second.set_value(std::move(result));
    }
}

std::expected<gtsam::Values, std::string> Isam2Worker::optimizeBatch(
    const gtsam::LevenbergMarquardtParams& lm_params) {
    try {
        // getFactorsUnsafe() is the full graph iSAM2 has accumulated
        // (priors + IMU + between + the current smart factors). With
        // findUnusedFactorSlots it may contain nullptr holes where factors
        // were removed; NonlinearFactorGraph::error()/linearize() skip null
        // factors, so LM handles them. Seed from the incremental estimate.
        const gtsam::NonlinearFactorGraph& graph = isam_->getFactorsUnsafe();
        const gtsam::Values initial = isam_->calculateEstimate();

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, lm_params);
        const double error_before = optimizer.error();
        gtsam::Values optimized = optimizer.optimize();
        const double error_after = optimizer.error();
        const auto t1 = Clock::now();

        spdlog::info(
            LOG_ID
            " optimizeBatch: {} iters, error {:.3f} -> {:.3f}, "
            "{} factors, {} values, {:.1f} ms",
            optimizer.iterations(), error_before, error_after, graph.size(),
            optimized.size(),
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        return optimized;
    } catch (const std::exception& e) {
        return std::unexpected(std::format("optimizeBatch threw [{}]: {}",
                                           typeid(e).name(), e.what()));
    }
}
