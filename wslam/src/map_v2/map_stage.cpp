#include "map_stage.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "associate_landmarks.hpp"
#include "build_factors.hpp"
#include "compute/stage.hpp"
#include "feed_loop_closure.hpp"
#include "filter_keyframe.hpp"
#include "loop_closure/merge.hpp"
#include "map/utils.hpp"
#include "map_common.hpp"
#include "submit_updates.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;
namespace utils = wslam::map::utils;

#define LOG_ID "[Mapping v2 stage]"

namespace {
// Moves the smoother's final snapshot to MapSnapshotVioName and publishes
// its loop-corrected copy under MapSnapshotName, with the loop closure report
std::optional<std::string> MergeLoopClosure(
    AnyBag& storage, const SharedBindings& shared,
    const loop_closure::LoopClosureWorker& worker) {
    auto vio = storage.get<MapSnapshot>(ResourceIdentifier::MapSnapshotName);
    if (!vio) {
        return "no final snapshot to merge loop closure into";
    }

    const auto anchor_of
        = [&shared](uint64_t landmark) -> std::optional<uint64_t> {
        const auto it
            = shared.landmark_anchor.find(wslam::map::LandmarkId{landmark});
        if (it == shared.landmark_anchor.end()) {
            return std::nullopt;
        }
        return it->second.v;
    };

    loop_closure::MergeStats stats;
    MapSnapshot corrected = loop_closure::ApplyCorrections(
        vio.value(), worker.graph(), anchor_of, &stats);
    LoopClosureReport report = loop_closure::MakeReport(worker);

    spdlog::info(LOG_ID
                 " loop closure ({}): {} keyframes, {} nodes, {} queries, {} "
                 "loops; corrected {} keyframes and {} landmarks ({} without "
                 "an anchor)",
                 report.model, report.keyframes, report.nodes, report.queries,
                 report.loops.size(), stats.keyframes_corrected,
                 stats.landmarks_corrected, stats.landmarks_unanchored);

    storage.set(ResourceIdentifier::MapSnapshotVioName, std::move(vio).value());
    storage.set(ResourceIdentifier::MapSnapshotName, std::move(corrected));
    storage.set(ResourceIdentifier::LoopClosureReportName, std::move(report));

    return std::nullopt;
}
}  // namespace

MappingStage v2::mapping::CreateMappingStage(
    compute::Compute& compute, AnyBag& storage, const WslamConfig& config,
    std::shared_ptr<loop_closure::LoopClosureWorker> loop_closure) {
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
        storage, *shared,
        BuildFactorsPass::Opts{
            .gate_min_rotation_rad = filter_config.min_rotation_rad,
            .gate_min_parallax_px = filter_config.min_parallax_px,
        }));
    stage->add_pass(std::move(submit));

    if (!loop_closure) {
        return MappingStage{
            .stage = std::move(stage),
            .flush_async
            = [shared, submit_ptr]() -> std::optional<std::string> {
                return submit_ptr->flush();
            },
        };
    }

    auto feed
        = std::make_unique<FeedLoopClosurePass>(storage, *shared, loop_closure);
    FeedLoopClosurePass* feed_ptr = feed.get();
    stage->add_pass(std::move(feed));

    return MappingStage{
        .stage = std::move(stage),
        .flush_async = [shared, submit_ptr, feed_ptr, loop_closure,
                        &storage]() -> std::optional<std::string> {
            if (auto err = submit_ptr->flush()) {
                return err;
            }

            feed_ptr->flush();
            if (auto err = loop_closure->finish()) {
                return "loop closure: " + std::move(err).value();
            }

            return MergeLoopClosure(storage, *shared, *loop_closure);
        },
    };
}
