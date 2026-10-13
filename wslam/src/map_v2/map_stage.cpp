#include "map_stage.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "associate_landmarks.hpp"
#include "build_factors.hpp"
#include "compute/stage.hpp"
#include "filter_keyframe.hpp"
#include "map/utils.hpp"
#include "map_common.hpp"
#include "submit_updates.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;
namespace utils = wslam::map::utils;

#define LOG_ID "[Mapping v2 stage]"

MappingStage v2::mapping::CreateMappingStage(compute::Compute& compute,
                                             AnyBag& storage,
                                             const WslamConfig& config) {
    spdlog::info(LOG_ID " Constructing mapping stage");

    // Owned here; the passes hold it by reference. flush_async keeps it alive
    // as long as Compute holds the passes
    auto shared = std::make_shared<SharedBindings>();

    const FilterKeyframeConfig filter_config{};

    SubmitSmootherPass::Opts submit_opts{};
    // Headless runs only read the final snapshot (flush() publishes it); the
    // GUI would read one every keyframe
    submit_opts.snapshot_every_n_keyframes
        = config.enable_gui ? 1U : utils::EnvU32("WSLAM_SNAPSHOT_EVERY_N", 0U);
    submit_opts.smoother.window_keyframes = utils::EnvU32(
        "WSLAM_SMOOTHER_WINDOW", submit_opts.smoother.window_keyframes);
    submit_opts.smoother.max_iterations = utils::EnvU32(
        "WSLAM_SMOOTHER_MAX_ITERS", submit_opts.smoother.max_iterations);

    spdlog::info(LOG_ID
                 " back-end config: window={} keyframes, max_iterations={}, "
                 "snapshot_every_n={}",
                 submit_opts.smoother.window_keyframes,
                 submit_opts.smoother.max_iterations,
                 submit_opts.snapshot_every_n_keyframes);

    auto stage
        = std::make_unique<compute::Stage>("Mapping v2", &compute.getPerf());

    auto submit
        = std::make_unique<SubmitSmootherPass>(storage, *shared, submit_opts);
    // The drain pass and the flushable reach the submit pass through this
    // pointer; all three live inside Compute together
    SubmitSmootherPass* submit_ptr = submit.get();

    stage->add_pass(
        std::make_unique<FilterKeyframePass>(storage, *shared, filter_config));
    stage->add_pass(std::make_unique<DrainSmootherPass>(*submit_ptr));
    stage->add_pass(std::make_unique<AssociateLandmarksPass>(
        storage, *shared,
        AssociateLandmarksPass::Opts{
            .association_tolerance_px = filter_config.association_tolerance_px,
        }));
    stage->add_pass(std::make_unique<BuildFactorsPass>(
        storage, *shared, BuildFactorsPass::Opts{}));
    stage->add_pass(std::move(submit));

    return MappingStage{
        .stage = std::move(stage),
        .flush_async = [shared, submit_ptr]() -> std::optional<std::string> {
            return submit_ptr->flush();
        },
    };
}
