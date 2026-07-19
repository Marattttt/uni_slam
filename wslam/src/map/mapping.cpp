#include "mapping.hpp"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "factor_builder.hpp"
#include "isam2_update.hpp"
#include "keyframe_gate.hpp"

using namespace wslam;

#define LOG_ID "[Mapping stage]"

namespace {

// Env-var overrides for the mapping back-end's performance/accuracy levers.
// Defaults live in Isam2UpdatePass::Opts; these let a benchmark sweep flip
// each knob without recompiling. See benchmarks/HOWTO_REEVALUATE.
const char* envOrNull(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

double envDouble(const char* name, double fallback) {
    if (const char* value = envOrNull(name)) {
        try {
            return std::stod(value);
        } catch (...) {
            spdlog::warn(LOG_ID " ignoring non-numeric {}='{}'", name, value);
        }
    }
    return fallback;
}

int envInt(const char* name, int fallback) {
    if (const char* value = envOrNull(name)) {
        try {
            return std::stoi(value);
        } catch (...) {
            spdlog::warn(LOG_ID " ignoring non-integer {}='{}'", name, value);
        }
    }
    return fallback;
}

uint32_t envU32(const char* name, uint32_t fallback) {
    const int value = envInt(name, static_cast<int>(fallback));
    return value < 0 ? fallback : static_cast<uint32_t>(value);
}

bool envBool(const char* name, bool fallback) {
    if (const char* value = envOrNull(name)) {
        const std::string s(value);
        return s == "1" || s == "true" || s == "TRUE" || s == "on"
               || s == "ON";
    }
    return fallback;
}

bool envCholesky(bool fallback) {
    if (const char* value = envOrNull("WSLAM_ISAM_FACTORIZATION")) {
        const std::string s(value);
        if (s == "CHOLESKY" || s == "cholesky") {
            return true;
        }
        if (s == "QR" || s == "qr") {
            return false;
        }
        spdlog::warn(LOG_ID " ignoring unknown WSLAM_ISAM_FACTORIZATION='{}'",
                     value);
    }
    return fallback;
}

}  // namespace

MappingStage wslam::CreateMappingStage(compute::Compute& compute,
                                       AnyBag& storage,
                                       const WslamConfig& config) {
    spdlog::info(LOG_ID " Constructing mapping stage");

    auto state = std::make_shared<MappingState>();

    auto stage = std::make_unique<compute::Stage>("Mapping", &compute.getPerf());

    auto gate = std::make_unique<KeyframeGatePass>(state);
    gate->setStorage(storage);

    auto builder = std::make_unique<FactorBuilderPass>(state);
    builder->setStorage(storage);

    // The iSAM2 update pass needs to read the builder's per-frame factor /
    // value containers. Hold a raw pointer (lifetimes are tied: both live in
    // the same Stage, destroyed together).
    const FactorBuilderPass* builder_ptr = builder.get();

    // Headless runs have no consumer of intermediate snapshots — the map
    // export only reads the post-flush one — so skip the (O(map size))
    // snapshot during the loop entirely (0 = never; flush() still publishes
    // the final one). The GUI reads it every frame. Env-overridable so a
    // sweep can restore the old cadence for an apples-to-apples baseline.
    Isam2UpdatePass::Opts updater_opts{};
    updater_opts.snapshot_every_n_drains
        = config.enable_gui ? 1U : envU32("WSLAM_SNAPSHOT_EVERY_N", 0U);

    // Live-loop iSAM2 levers + final batch optimisation. Code defaults are
    // the chosen configuration; each knob is env-overridable for benchmark
    // sweeps (see benchmarks/HOWTO_REEVALUATE).
    updater_opts.relinearize_threshold = envDouble(
        "WSLAM_ISAM_RELIN_THRESH", updater_opts.relinearize_threshold);
    updater_opts.relinearize_skip
        = envInt("WSLAM_ISAM_RELIN_SKIP", updater_opts.relinearize_skip);
    updater_opts.use_cholesky = envCholesky(updater_opts.use_cholesky);
    updater_opts.final_batch_optimize
        = envBool("WSLAM_FINAL_BATCH", updater_opts.final_batch_optimize);
    updater_opts.final_batch_max_iterations = envInt(
        "WSLAM_FINAL_BATCH_ITERS", updater_opts.final_batch_max_iterations);

    spdlog::info(LOG_ID
                 " back-end config: relin_thresh={}, relin_skip={}, "
                 "factorization={}, snapshot_every_n={}, final_batch={} "
                 "(max_iters={})",
                 updater_opts.relinearize_threshold,
                 updater_opts.relinearize_skip,
                 updater_opts.use_cholesky ? "CHOLESKY" : "QR",
                 updater_opts.snapshot_every_n_drains,
                 updater_opts.final_batch_optimize,
                 updater_opts.final_batch_max_iterations);

    auto updater = std::make_unique<Isam2UpdatePass>(std::move(state),
                                                     *builder_ptr, updater_opts);
    updater->setStorage(storage);
    // Same lifetime argument as for builder_ptr above: the updater is added
    // to `stage` below and lives as long as that stage does.
    Isam2UpdatePass* updater_ptr = updater.get();

    // Drain pass goes FIRST so the keyframe gate and factor builder see
    // up-to-date pose / smart-factor estimates from the previous iSAM
    // round. Without this, the factor builder would read stale
    // smart_factor_indices and schedule removal of factor indices iSAM
    // had already replaced — yielding VariableIndex::remove failures.
    auto drainer = std::make_unique<Isam2DrainPass>(*updater_ptr);

    stage->add_pass(std::move(drainer));
    stage->add_pass(std::move(gate));
    stage->add_pass(std::move(builder));
    stage->add_pass(std::move(updater));

    return MappingStage{
        .stage = std::move(stage),
        .flush_async =
            [updater_ptr]() { return updater_ptr->flush(); },
    };
}
