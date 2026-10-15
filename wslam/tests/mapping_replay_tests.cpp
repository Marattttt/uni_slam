// Replays a recorded v2 front end (frontend_recording.hpp) through the CPU
// RANSAC and triangulation passes and the v2 mapping stage: the mapping stage
// tested on real data in seconds, without the GPU or the model.
//
// WSLAM_FRONTEND_RECORDING   recording to replay (the test skips without it),
//                            made by `pc_wslam -v2 --record-frontend=PATH`
// WSLAM_REPLAY_MAP_OUT       optional .ply path to export the replayed map to,
//                            for benchmarks/compare.py
// WSLAM_PREFILTER_*          frame prefilter knobs PrefilterPredictsTheGate
//                            scores (see frame_prefilter.hpp)

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
#include <vector>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/pass.hpp"
#include "compute/stage.hpp"
#include "data/provider_base.hpp"
#include "export.hpp"
#include "feature_detect_v2/frame_prefilter.hpp"
#include "feature_detect_v2/motion_prediction.hpp"
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

// Runs the frame prefilter's prediction on every replayed frame after the
// second keyframe and scores it against the keyframe gate's decision on the
// same frame: a frame predicted small that the gate accepted would have been
// skipped wrongly (the replay itself skips nothing, so the run follows the
// golden keyframes). Tune the WSLAM_PREFILTER_* knobs with it; the summary
// line reports the share of frames the prefilter would skip
TEST(MappingReplayTest, PrefilterPredictsTheGate) {
    const auto recording_path = EnvPath("WSLAM_FRONTEND_RECORDING");
    if (!recording_path) {
        GTEST_SKIP() << "WSLAM_FRONTEND_RECORDING is not set";
    }

    auto recording = v2::LoadFrontendRecording(*recording_path);
    ASSERT_TRUE(recording.has_value()) << recording.error();
    const size_t frame_count = recording->frames.size();
    const data::CamSensorParams camera = recording->camera;
    const data::IMUSensorParams imu_params = recording->imu_params;

    compute::Compute compute;
    AnyBag& storage = compute.getStorage();

    auto replay
        = std::make_shared<v2::FrontendReplay>(storage, *std::move(recording));
    replay->publishSensorParams();

    auto pose_stage = std::make_unique<compute::Stage>("Replayed pose estimate",
                                                       &compute.getPerf());
    pose_stage->add_pass(std::make_unique<compute::CustomPass>(
        "[Replay front end]",
        [replay](compute::CustomPass*) { return replay->publishFrame(); }));
    pose_stage->add_pass(std::make_unique<v2::RansacPass>(storage));
    pose_stage->add_pass(std::make_unique<v2::TriangulatePass>(storage));
    compute.addStage(std::move(pose_stage));

    v2::MotionPredictor predictor{
        camera, imu_params, v2::ReadFramePrefilterOptionsFromEnv().prediction};

    struct Scores {
        size_t predicted = 0;
        size_t small = 0;
        size_t accepted = 0;
        size_t false_skips = 0;
        // Largest measured / upper bound over the frames the gate measured:
        // above 1 the bound was too tight for that frame
        double worst_parallax_ratio = 0.0;
        double worst_rotation_ratio = 0.0;
        double parallax_error_sum = 0.0;
        size_t measured = 0;
    };
    Scores scores;
    std::optional<v2::KeyframeMotionPrediction> current;

    // Between the pose estimate and the mapping stage, like the prefilter's
    // prediction reaching the gate
    compute.addStage(MakeStage(
        compute, "Predict motion",
        [&](compute::CustomPass*) -> std::optional<std::string> {
            current.reset();
            const auto prior = storage.getPtr<v2::KeyframeMotionPrior>(
                ResourceIdentifier::KeyframeMotionPriorName);
            const auto frame_ts = storage.get<uint64_t>(
                ResourceIdentifier::FrameTimestampNsName);
            const auto imu = storage.getPtr<std::vector<data::IMUReading>>(
                ResourceIdentifier::GetImuVecName());
            if (!prior || !frame_ts || !imu) {
                return std::nullopt;
            }

            auto prediction = predictor.predict(*prior.value(), *imu.value(),
                                                frame_ts.value());
            if (!prediction) {
                return "predicting: " + prediction.error();
            }
            ++scores.predicted;
            scores.small += prediction->small_motion ? 1 : 0;
            current = prediction.value();

            // The gate's measurements of the same frame
            const auto triangulation
                = storage.getPtr<v2::KeypointTriangulationResult>(
                    ResourceIdentifier::KeypointTriangulationName);
            const auto ransac = storage.getPtr<v2::KeypointRansacResult>(
                ResourceIdentifier::KeypointRansacName);
            if (triangulation && ransac && !ransac.value()->inliers.empty()) {
                std::vector<double> parallaxes;
                for (const auto& inlier : ransac.value()->inliers) {
                    parallaxes.push_back(
                        (inlier.curr_px - inlier.prev_px).norm());
                }
                const auto middle
                    = parallaxes.begin()
                      + static_cast<std::ptrdiff_t>(parallaxes.size() / 2);
                std::nth_element(parallaxes.begin(), middle, parallaxes.end());
                const double rotation = map::util::ComputeRotationAngle(
                    triangulation.value()->rotation);

                ++scores.measured;
                scores.parallax_error_sum
                    += std::abs(*middle - prediction->parallax_px);
                scores.worst_parallax_ratio
                    = std::max(scores.worst_parallax_ratio,
                               *middle / prediction->parallax_upper_px);
                scores.worst_rotation_ratio
                    = std::max(scores.worst_rotation_ratio,
                               rotation / prediction->rotation_upper_rad);
            }

            storage.set(ResourceIdentifier::KeyframeMotionPredictionName,
                        prediction.value());
            return std::nullopt;
        }));

    WslamConfig config{};
    config.use_pipeline_v2 = true;
    auto mapping = v2::mapping::CreateMappingStage(compute, storage, config);
    compute.addStage(std::move(mapping.stage));

    // Before the decision check consumes the reference marker
    compute.addStage(MakeStage(
        compute, "Score prediction",
        [&](compute::CustomPass*) -> std::optional<std::string> {
            const bool accepted
                = storage
                      .get<bool>(
                          ResourceIdentifier::FeatureReferenceAdvanceName)
                      .value_or(false);
            if (!current || !accepted) {
                return std::nullopt;
            }
            ++scores.accepted;
            if (current->small_motion) {
                ++scores.false_skips;
                spdlog::warn(
                    "[Prefilter replay] accepted keyframe predicted small: dt "
                    "{:.3f} s, rotation <= {:.4f} rad, parallax <= {:.2f} px",
                    current->dt_s, current->rotation_upper_rad,
                    current->parallax_upper_px);
            }
            return std::nullopt;
        }));

    compute.addStage(MakeStage(
        compute, "Check keyframe decision",
        [replay](compute::CustomPass*) { return replay->checkDecision(); }));

    ASSERT_EQ(compute.initizalize(), std::nullopt);

    for (size_t frame = 0; frame < frame_count; ++frame) {
        const auto err = compute.execute();
        ASSERT_FALSE(err.has_value()) << "frame " << frame << ": " << *err;
        ASSERT_FALSE(replay->divergence().has_value())
            << "the replay diverged at frame " << frame;
    }
    ASSERT_EQ(mapping.flush_async(), std::nullopt);

    spdlog::info(
        "[Prefilter replay] {} frames, {} predicted: {} predicted small "
        "({:.1f}% "
        "of the predicted frames), {} keyframes accepted, {} of them predicted "
        "small; mean |parallax error| {:.2f} px, worst measured/upper parallax "
        "{:.2f}, rotation {:.2f}",
        frame_count, scores.predicted, scores.small,
        scores.predicted > 0 ? 100.0 * static_cast<double>(scores.small)
                                   / static_cast<double>(scores.predicted)
                             : 0.0,
        scores.accepted, scores.false_skips,
        scores.measured > 0
            ? scores.parallax_error_sum / static_cast<double>(scores.measured)
            : 0.0,
        scores.worst_parallax_ratio, scores.worst_rotation_ratio);

    EXPECT_GT(scores.predicted, 0U);
    EXPECT_EQ(scores.false_skips, 0U);
}

}  // namespace
