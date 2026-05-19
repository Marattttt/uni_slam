#include "isam2_worker.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <format>
#include <typeinfo>
#include <utility>

using namespace wslam;

#define LOG_ID "[ISAM2 worker]"

Isam2Worker::Isam2Worker() {
    gtsam::ISAM2Params params;
    params.optimizationParams = gtsam::ISAM2GaussNewtonParams();
    // Relinearise + re-eliminate the Bayes tree every N updates rather than
    // every frame. The per-update cost of N==1 is dominated by re-elim of
    // the entire affected subtree; batching to N==10 amortises that across
    // many keyframes for an order-of-magnitude speedup in steady state.
    // Map quality is essentially unaffected because the threshold below
    // still gates which variables actually move when relin does fire.
    params.relinearizeSkip = 10;
    // Variable is queued for relinearisation when its delta exceeds this
    // value. Raised from 0.01 so a smaller set of variables qualifies on
    // each relin cycle; slightly slower convergence for variables that
    // drift, but a meaningful per-update cost reduction.
    params.relinearizeThreshold = 0.05;
    isam_ = std::make_unique<gtsam::ISAM2>(params);

    thread_ = std::thread([this] { run(); });
    spdlog::info(LOG_ID " Worker thread started");
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
        // Caller contract: drain the previous future before submitting a
        // new one. Violating this would silently drop work.
        assert(!pending_.has_value()
               && "Isam2Worker: previous submission still pending");
        pending_.emplace(std::move(work), std::move(p));
    }
    cv_.notify_one();
    return fut;
}

void Isam2Worker::run() {
    while (true) {
        std::optional<std::pair<Work, std::promise<Result>>> item;
        {
            std::unique_lock<std::mutex> lk(mu_);
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
                          " frame={}: submitting factors={}, new_values={}",
                          item->first.frame_id,
                          item->first.new_factors.size(),
                          item->first.new_values.size());
            const auto r = isam_->update(item->first.new_factors,
                                         item->first.new_values);
            for (uint32_t i = 0; i < item->first.extra_updates; ++i) {
                isam_->update();
            }
            result.latest_values = isam_->calculateEstimate();
            result.factor_count = isam_->getFactorsUnsafe().size();
            spdlog::info(
                LOG_ID
                " frame={} OK: relin={}, factors_recalc={}, total_factors={}, "
                "total_values={}",
                item->first.frame_id, r.variablesRelinearized,
                r.factorsRecalculated, result.factor_count,
                result.latest_values.size());
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
