#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "anybag.hpp"
#include "compute/compute.hpp"
#include "compute/stage.hpp"
#include "map_build_factors.hpp"
#include "map_common.hpp"
#include "map_filter_keyframe.hpp"
#include "map_submit_updates.hpp"
#include "mapping.hpp"
#include "utils.hpp"

using namespace wslam;

#define LOG_ID "[Mapping stage]"

namespace utils = wslam::map::utils;

MappingStage wslam::CreateMappingStage(compute::Compute& compute,
                                       AnyBag& storage,
                                       const WslamConfig& config) {
    spdlog::info(LOG_ID " Constructing mapping stage");

    // Persistent cross-pass state (iSAM2-adjacent bookkeeping, calibration,
    // landmark tracks). Owned here by shared_ptr; the passes hold it by
    // reference. Kept alive by the flush_async capture below for as long as the
    // passes (which move into Compute) live — see the note there.
    auto bindings = std::make_shared<map::MappingSharedBindings>();

    auto stage
        = std::make_unique<compute::Stage>("Mapping", &compute.getPerf());

    auto filter = std::make_unique<map::FilterKeyframePass>(compute, *bindings);

    // Headless runs have no consumer of intermediate snapshots — the map export
    // only reads the post-flush one — so skip the (O(map size)) snapshot during
    // the loop entirely (0 = never; flush() still publishes the final one). The
    // GUI reads it every frame. Env-overridable so a sweep can restore the old
    // cadence for an apples-to-apples baseline.
    map::SubmitUpdatesPass::Opts submit_opts{};
    submit_opts.snapshot_every_n_keyframes
        = config.enable_gui ? 1U : utils::EnvU32("WSLAM_SNAPSHOT_EVERY_N", 0U);

    // Live-loop iSAM2 levers. Code defaults are the chosen configuration; each
    // knob is env-overridable for benchmark sweeps (see HOWTO_REEVALUATE).
    submit_opts.relinearize_threshold = utils::EnvDouble(
        "WSLAM_ISAM_RELIN_THRESH", submit_opts.relinearize_threshold);
    submit_opts.use_cholesky = utils::EnvCholesky(submit_opts.use_cholesky);
    submit_opts.extra_updates
        = utils::EnvU32("WSLAM_ISAM_EXTRA_UPDATES", submit_opts.extra_updates);

    spdlog::info(LOG_ID
                 " back-end config: relin_thresh={}, "
                 "factorization={}, snapshot_every_n={}, extra_updates={}",
                 submit_opts.relinearize_threshold,
                 submit_opts.use_cholesky ? "CHOLESKY" : "QR",
                 submit_opts.snapshot_every_n_keyframes,
                 submit_opts.extra_updates);

    auto submit = std::make_unique<map::SubmitUpdatesPass>(storage, *bindings,
                                                           submit_opts);
    // The drain pass delegates to submit's public drainPending(); it lives in
    // the same Stage (destroyed together), so a raw reference is safe.
    map::SubmitUpdatesPass* submit_ptr = submit.get();
    auto drain = std::make_unique<map::DrainUpdatesPass>(*submit_ptr);

    auto build = std::make_unique<map::BuildFactorsPass>(
        storage, *bindings, map::BuildFactorsPass::Opts{});

    // Stage order matters:
    //   Filter -> Drain -> Build -> Submit
    // Drain goes *after* the filter (which stops the stage on rejected frames)
    // so it only runs on accepted keyframes, letting the inter-keyframe gap
    // overlap the async iSAM solve; and *before* Build so the builder reads the
    // drained keyframe's smart-factor indices, optimised poses, and propagated
    // velocity/bias. Submit hands the built bundle to the worker without
    // blocking; its result is drained on the next accepted keyframe.
    stage->add_pass(std::move(filter));
    stage->add_pass(std::move(drain));
    stage->add_pass(std::move(build));
    stage->add_pass(std::move(submit));

    return MappingStage{
        .stage = std::move(stage),
        // Capture `bindings` (shared_ptr) purely to extend its lifetime: the
        // stage's passes reference *bindings and are moved into Compute, while
        // this flushable is also held by Compute, so the state outlives the
        // passes. `submit_ptr` stays valid for the same reason.
        .flush_async = [bindings, submit_ptr]() -> std::optional<std::string> {
            return submit_ptr->flush();
        },
    };
}
