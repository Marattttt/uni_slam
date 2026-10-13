// Replays a recorded v2 front end (frontend_recording.hpp) through the CPU
// RANSAC and triangulation passes and the v2 mapping stage: the mapping stage
// tested on real data in seconds, without the GPU or the model.
//
// WSLAM_FRONTEND_RECORDING   recording to replay (the test skips without it),
//                            made by `pc_wslam -v2 --record-frontend=PATH`
// WSLAM_REPLAY_MAP_OUT       optional .ply path to export the replayed map to,
//                            for benchmarks/compare.py

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/pass.hpp"
#include "compute/stage.hpp"
#include "export.hpp"
#include "frontend_recording.hpp"
#include "map/map_helpers_lods.hpp"
#include "map_v2/map_stage.hpp"
#include "models.hpp"
#include "pose_estimate_v2/ransac.hpp"
#include "pose_estimate_v2/triangulate.hpp"

namespace {

using namespace wslam;

std::optional<std::filesystem::path> EnvPath(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path(value);
}

std::unique_ptr<compute::Stage> MakeStage(compute::Compute& compute,
                                          const std::string& name,
                                          compute::CustomPass::Callback body) {
    auto stage = std::make_unique<compute::Stage>(name, &compute.getPerf());
    stage->add_pass(std::make_unique<compute::CustomPass>(
        std::format("[{}]", name), std::move(body)));
    return stage;
}

TEST(MappingReplayTest, ReplaysTheGoldenRun) {
    const auto recording_path = EnvPath("WSLAM_FRONTEND_RECORDING");
    if (!recording_path) {
        GTEST_SKIP() << "WSLAM_FRONTEND_RECORDING is not set";
    }

    auto recording = v2::LoadFrontendRecording(*recording_path);
    ASSERT_TRUE(recording.has_value()) << recording.error();
    ASSERT_FALSE(recording->frames.empty());
    const size_t frame_count = recording->frames.size();

    // The CPU stages need no GPU, so Compute is not prepared
    compute::Compute compute;
    AnyBag& storage = compute.getStorage();

    auto replay
        = std::make_shared<v2::FrontendReplay>(storage, *std::move(recording));
    replay->publishSensorParams();

    // Stands in for the feature detect stage and the model-readback pass of
    // the pose estimate stage
    auto pose_stage = std::make_unique<compute::Stage>("Replayed pose estimate",
                                                       &compute.getPerf());
    pose_stage->add_pass(std::make_unique<compute::CustomPass>(
        "[Replay front end]",
        [replay](compute::CustomPass*) { return replay->publishFrame(); }));
    pose_stage->add_pass(std::make_unique<v2::RansacPass>(storage));
    pose_stage->add_pass(std::make_unique<v2::TriangulatePass>(storage));
    compute.addStage(std::move(pose_stage));

    WslamConfig config{};
    config.use_pipeline_v2 = true;
    auto mapping = v2::mapping::CreateMappingStage(compute, storage, config);
    compute.addStage(std::move(mapping.stage));

    compute.addStage(MakeStage(
        compute, "Check keyframe decision",
        [replay](compute::CustomPass*) { return replay->checkDecision(); }));

    ASSERT_EQ(compute.initizalize(), std::nullopt);

    for (size_t frame = 0; frame < frame_count; ++frame) {
        const auto err = compute.execute();
        ASSERT_FALSE(err.has_value()) << "frame " << frame << ": " << *err;
        if (replay->divergence()) {
            break;
        }
    }

    ASSERT_EQ(mapping.flush_async(), std::nullopt);

    if (const auto& divergence = replay->divergence()) {
        FAIL() << std::format(
            "the mapping stage diverged from the recording at frame {} of {}: "
            "recorded reference advance {}, replayed {}. Re-record the front "
            "end after changing the keyframe gate",
            divergence->frame_index, frame_count, divergence->recorded_advance,
            divergence->replayed_advance);
    }
    EXPECT_EQ(replay->framesReplayed(), frame_count);

    const auto snapshot
        = storage.getPtr<MapSnapshot>(ResourceIdentifier::MapSnapshotName);
    ASSERT_TRUE(snapshot.has_value());
    const MapSnapshot& map = *snapshot.value();

    const auto& golden = replay->recording().golden_keyframes;
    spdlog::info(
        "[Mapping replay] {} frames: {} keyframes ({} in the golden run), {} "
        "landmarks ({} in the golden run)",
        frame_count, map.keyframes.size(), golden.size(), map.landmarks.size(),
        replay->recording().golden_landmark_count);

    EXPECT_FALSE(map.keyframes.empty());
    EXPECT_FALSE(map.landmarks.empty());

    // Same inputs, same single-threaded optimiser: the replay reproduces the
    // golden map up to floating-point noise
    ASSERT_EQ(map.keyframes.size(), golden.size());
    EXPECT_EQ(map.landmarks.size(), replay->recording().golden_landmark_count);
    double max_translation = 0.0;
    double max_rotation = 0.0;
    for (size_t i = 0; i < golden.size(); ++i) {
        ASSERT_EQ(map.keyframes[i].id.v, golden[i].id.v);
        EXPECT_EQ(map.keyframes[i].timestamp_ns, golden[i].timestamp_ns);
        max_translation = std::max(
            max_translation,
            (map.keyframes[i].t_world_cam - golden[i].t_world_cam).norm());
        max_rotation = std::max(max_rotation,
                                map::util::ComputeRotationAngle(
                                    map.keyframes[i].R_world_cam.transpose()
                                    * golden[i].R_world_cam));
    }
    spdlog::info(
        "[Mapping replay] largest deviation from the golden run: {:.3g} m, "
        "{:.3g} rad",
        max_translation, max_rotation);
    EXPECT_LT(max_translation, 1e-6);
    EXPECT_LT(max_rotation, 1e-6);

    if (const auto map_out = EnvPath("WSLAM_REPLAY_MAP_OUT")) {
        EXPECT_EQ(ExportMap(storage, ExportOpts{.map_path = *map_out}),
                  std::nullopt);
    }
}

}  // namespace
