#include "smoother_worker.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <exception>
#include <format>
#include <typeinfo>
#include <utility>

#include "assert.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;

#define LOG_ID "[Smoother worker]"

LocalSmoother::Result LocalSmoother::update(
    const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values, const KeyTimestampMap& timestamps,
    const std::set<size_t>& remove_slots) {
    for (const size_t slot : remove_slots) {
        WSLAM_ASSERT(slot < factors_.size() && factors_.at(slot) != nullptr,
                     "removing empty smoother slot {}", slot);
    }
    removeFactors(remove_slots);
    return gtsam::BatchFixedLagSmoother::update(new_factors, new_values,
                                                timestamps);
}

namespace {
gtsam::LevenbergMarquardtParams MakeOptimizerParams(
    const SmootherWorker::Params& params) {
    gtsam::LevenbergMarquardtParams optimizer;
    optimizer.setMaxIterations(static_cast<int>(params.max_iterations));
    return optimizer;
}
}  // namespace

SmootherWorker::SmootherWorker(Params params)
    : params_(params),
      smoother_(std::make_unique<LocalSmoother>(
          // A key stamped t is marginalised once the newest key's stamp
          // exceeds t + lag, keeping window_keyframes keyframes
          static_cast<double>(params.window_keyframes) - 1.0,
          MakeOptimizerParams(params), params.enforce_consistency)),
      worker_([smoother = smoother_.get()](Work&& work) {
          return Run(*smoother, std::move(work));
      }) {
    WSLAM_ASSERT(params.window_keyframes >= 2,
                 "a window of {} keyframes can't hold an IMU edge",
                 params.window_keyframes);
    spdlog::info(LOG_ID
                 " Worker thread started (window={} keyframes, "
                 "max_iterations={}, enforce_consistency={})",
                 params.window_keyframes, params.max_iterations,
                 params.enforce_consistency);
}

std::future<SmootherWorker::Result> SmootherWorker::submit(Work work) {
    return worker_.submit(std::move(work));
}

SmootherWorker::Result SmootherWorker::Run(LocalSmoother& smoother,
                                           Work&& work) {
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](Clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };

    Result result;
    const FactorBundle& bundle = work.bundle;
    try {
        spdlog::debug(LOG_ID " x{}: {} factors, {} values, {} removals",
                      work.keyframe_id, bundle.new_factors.size(),
                      bundle.new_values.size(), bundle.remove_slots.size());

        const auto update_begin = Clock::now();
        const auto update
            = smoother.update(bundle.new_factors, bundle.new_values,
                              bundle.timestamps, bundle.remove_slots);
        const auto update_end = Clock::now();

        result.latest_values = smoother.calculateEstimate();

        const gtsam::NonlinearFactorGraph& factors = smoother.getFactors();
        result.live_slots.reserve(factors.size());
        for (size_t slot = 0; slot < factors.size(); ++slot) {
            if (const auto& factor = factors.at(slot)) {
                result.live_slots.emplace(factor.get(), slot);
                ++result.factor_count;
            }
        }
        result.iterations = update.iterations;
        result.graph_error = update.error;

        spdlog::info(LOG_ID
                     " x{} OK: iterations={}, error={:.3f}, factors={} "
                     "(slots {}), values={}, update_ms={:.1f}",
                     work.keyframe_id, update.iterations, update.error,
                     result.factor_count, factors.size(),
                     result.latest_values.size(),
                     milliseconds(update_end - update_begin));
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
        result.error = std::format(
            "smoother update: indeterminate linear system at key '{}' (chr "
            "'{}' index {}). Likely cause: a variable has no constraints "
            "yet. what(): {}",
            gtsam::DefaultKeyFormatter(e.nearbyVariable()),
            static_cast<char>(gtsam::Symbol(e.nearbyVariable()).chr()),
            gtsam::Symbol(e.nearbyVariable()).index(), e.what());
    } catch (const std::exception& e) {
        result.error = std::format("smoother update threw [{}]: {}",
                                   typeid(e).name(), e.what());
    }

    if (result.error) {
        result.latest_values.clear();
        result.live_slots.clear();
    }
    return result;
}
