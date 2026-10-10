#include <gtest/gtest.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/pass.hpp"
#include "compute/stage.hpp"
#include "data/euroc_provider.hpp"
#include "data/provider_base.hpp"
#include "feature_detect/feature_detect.hpp"
#include "keyframe_schedule.hpp"
#include "map/map_helpers_lods.hpp"
#include "models.hpp"
#include "pose_estimate_cpu/pose_estimate_cpu.hpp"
#include "wslam.hpp"

// Runs the v1 and the v2 pose estimate stage over the keyframes v1 selected
// on EuRoC V1_01_easy, so each estimate spans two consecutive keyframes, and
// scores both against the ground truth. Prints a per-pair table and summary
// (also written to $WSLAM_POSE_REPORT_OUT when set). Needs the dataset, the
// shaders, the model and a GPU, so it skips without EUROC_DIR,
// WSLAM_SHADER_SRC_DIR or WSLAM_ONNX_MODEL

using namespace wslam;

namespace {

constexpr const char* kSchedulePath
    = WSLAM_TEST_DATA_DIR "/v1_keyframes_V101.json";
constexpr double kRadToDeg = 180.0 / std::numbers::pi;
// Translation direction is only scored for baselines this long; below it the
// direction is dominated by noise for any estimator
constexpr double kMinScoredBaselineM = 0.05;

struct PairEstimate {
    uint64_t prev_ts = 0;
    uint64_t curr_ts = 0;
    bool recovered = false;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    size_t matches = 0;
    size_t inliers = 0;
    size_t landmarks = 0;
    double mean_reprojection_px = 0.0;
};

struct PipelineRun {
    std::vector<PairEstimate> pairs;
    data::CamSensorParams camera;
    double seconds = 0.0;
};

// Reads the frame timestamp of the current iteration, so a recorder can pair
// it with the previous one
class PairRecorder {
   public:
    explicit PairRecorder(AnyBag& storage) : storage_(storage) {}

    // The pair the estimate in storage (if any) belongs to. Call once per
    // frame
    std::optional<PairEstimate> nextPair() {
        const auto timestamp
            = storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName);
        std::optional<PairEstimate> pair;
        if (previous_ts_ && timestamp) {
            pair
                = PairEstimate{.prev_ts = *previous_ts_, .curr_ts = *timestamp};
        }
        previous_ts_ = timestamp;
        return pair;
    }

   private:
    AnyBag& storage_;
    std::optional<uint64_t> previous_ts_;
};

void FillFromTriangulation(PairEstimate& pair, const auto& triangulation) {
    pair.recovered = triangulation.stats.pose_recovered;
    pair.rotation = triangulation.rotation;
    pair.translation = triangulation.translation;
    pair.landmarks = triangulation.stats.landmark_count;
    pair.mean_reprojection_px = triangulation.stats.mean_reprojection_error_px;
}

std::unique_ptr<compute::Stage> MakeStage(compute::Compute& compute,
                                          std::string name,
                                          compute::CustomPass::Callback body) {
    auto stage = std::make_unique<compute::Stage>(name, &compute.getPerf());
    stage->add_pass(std::make_unique<compute::CustomPass>(
        std::format("[{}]", name), std::move(body)));
    return stage;
}

std::optional<std::string> Execute(compute::Compute& compute,
                                   size_t iterations) {
    if (auto err = compute.initizalize()) {
        return "initializing: " + std::move(err).value();
    }
    // One iteration per keyframe. Compute::execute reports the end of the
    // stream as success, so it cannot tell when to stop
    for (size_t i = 0; i < iterations; ++i) {
        if (auto err = compute.execute()) {
            return std::format("iteration {}: {}", i, std::move(err).value());
        }
    }
    return std::nullopt;
}

std::expected<data::CamSensorParams, std::string> Camera(
    const compute::Compute& compute) {
    auto camera = compute.getStorage().get<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    if (!camera) {
        return std::unexpected("no camera parameters");
    }
    return std::move(camera).value();
}

std::expected<PipelineRun, std::string> RunV1(
    const KeyframeSchedule& schedule) {
    const auto start = std::chrono::steady_clock::now();

    data::EurocProvider euroc(data::CreateEurocProviderOpts());
    compute::Compute compute;
    if (auto err = compute.prepare(compute::createDefaultConfig())) {
        return std::unexpected("preparing compute: " + std::move(err).value());
    }

    GpuSharedBindings shared(compute.getGPUPtr(), compute.getStorage());
    if (auto err = shared.initialize()) {
        return std::unexpected("shared bindings: " + std::move(err).value());
    }
    if (auto err = impl::SetSensorIntrinsics(compute, euroc)) {
        return std::unexpected("intrinsics: " + std::move(err).value());
    }

    const WslamConfig config{};
    compute.addStage(MakeStage(compute, "Clear bindings",
                               [gpu = compute.getGPUPtr()](auto*) {
                                   return gpu->clearBuffersAndOffsets();
                               }));
    compute.addStage(CreateFeatureDetectStage(
        compute, shared, data::AdaptProvider<1U>(euroc), "features", config,
        MakeKeyframeFilter(schedule)));
    compute.addStage(
        CreatePoseEstimateCPUStage(compute, shared, "features", config));

    PipelineRun run;
    AnyBag& storage = compute.getStorage();
    compute.addStage(MakeStage(
        compute, "Record v1",
        [&storage, &run, recorder = PairRecorder(storage)](
            auto*) mutable -> std::optional<std::string> {
            auto pair = recorder.nextPair();
            auto triangulation = storage.take<TriangulationResult>(
                ResourceIdentifier::TriangulationResultName);
            auto ransac = storage.take<RansacResult>(
                ResourceIdentifier::RansacResultName);
            if (pair && triangulation && ransac) {
                FillFromTriangulation(*pair, *triangulation);
                pair->matches = ransac->stats.total_matches;
                pair->inliers = ransac->stats.inlier_count;
                run.pairs.push_back(*pair);
            }
            // Without the mapping stage nothing advances the reference
            // feature set; advancing it every frame pairs consecutive
            // keyframes, like the v2 model does
            storage.set(ResourceIdentifier::FeatureReferenceAdvanceName, true);
            return std::nullopt;
        }));

    if (auto err = Execute(compute, schedule.frame_indices.size())) {
        return std::unexpected(std::move(err).value());
    }

    auto camera = Camera(compute);
    if (!camera) {
        return std::unexpected(std::move(camera).error());
    }
    run.camera = std::move(camera).value();
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()
                                                - start)
                      .count();
    return run;
}

std::expected<PipelineRun, std::string> RunV2(
    const KeyframeSchedule& schedule) {
    const auto start = std::chrono::steady_clock::now();

    data::EurocProvider euroc(data::CreateEurocProviderOpts());
    compute::Compute compute;
    if (auto err = compute.prepare(compute::createDefaultConfig())) {
        return std::unexpected("preparing compute: " + std::move(err).value());
    }

    WslamConfig config{};
    config.use_pipeline_v2 = true;
    config.keyframes_path = kSchedulePath;
    if (auto err = CreateWslamPipelineV2(compute, euroc, config)) {
        return std::unexpected("creating pipeline: " + std::move(err).value());
    }

    PipelineRun run;
    AnyBag& storage = compute.getStorage();
    compute.addStage(MakeStage(
        compute, "Record v2",
        [&storage, &run, recorder = PairRecorder(storage)](
            auto*) mutable -> std::optional<std::string> {
            auto pair = recorder.nextPair();
            auto triangulation = storage.take<v2::KeypointTriangulationResult>(
                ResourceIdentifier::KeypointTriangulationName);
            auto ransac = storage.take<v2::KeypointRansacResult>(
                ResourceIdentifier::KeypointRansacName);
            if (pair && triangulation && ransac) {
                FillFromTriangulation(*pair, *triangulation);
                pair->matches = ransac->stats.total_matches;
                pair->inliers = ransac->stats.inlier_count;
                run.pairs.push_back(*pair);
            }
            return std::nullopt;
        }));

    if (auto err = Execute(compute, schedule.frame_indices.size())) {
        return std::unexpected(std::move(err).value());
    }

    auto camera = Camera(compute);
    if (!camera) {
        return std::unexpected(std::move(camera).error());
    }
    run.camera = std::move(camera).value();
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()
                                                - start)
                      .count();
    return run;
}

// Body (IMU) pose in the world frame from state_groundtruth_estimate0
struct GroundTruthSample {
    uint64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

std::expected<std::vector<GroundTruthSample>, std::string> LoadGroundTruth(
    const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        return std::unexpected(
            std::format("could not open '{}'", path.string()));
    }

    std::vector<GroundTruthSample> samples;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.starts_with('#')) {
            continue;
        }

        // timestamp, p_RS_R x y z, q_RS w x y z, ...
        constexpr size_t kColumns = 8;
        std::vector<std::string_view> fields;
        for (const auto field : std::views::split(line, ',')) {
            fields.emplace_back(field.begin(), field.end());
            if (fields.size() == kColumns) {
                break;
            }
        }
        if (fields.size() < kColumns) {
            return std::unexpected(std::format("short line '{}'", line));
        }

        GroundTruthSample sample{};
        std::array<double, kColumns - 1> values{};
        const auto parse = [](std::string_view text, auto& value) {
            const auto [end, error] = std::from_chars(
                text.data(), text.data() + text.size(), value);
            return error == std::errc{} && end == text.data() + text.size();
        };
        bool parsed = parse(fields[0], sample.timestamp);
        for (size_t i = 1; i < kColumns; ++i) {
            parsed = parsed && parse(fields[i], values.at(i - 1));
        }
        if (!parsed) {
            return std::unexpected(std::format("malformed line '{}'", line));
        }

        sample.position = {values[0], values[1], values[2]};
        sample.orientation
            = Eigen::Quaterniond(values[3], values[4], values[5], values[6])
                  .normalized();
        samples.push_back(sample);
    }

    if (samples.empty()) {
        return std::unexpected(
            std::format("'{}' has no samples", path.string()));
    }
    return samples;
}

// Camera pose in the world at `timestamp`, interpolated between the two
// ground truth samples around it. nullopt when either is over 10 ms away
std::optional<Eigen::Isometry3d> CameraPoseAt(
    const std::vector<GroundTruthSample>& samples, uint64_t timestamp,
    const Eigen::Isometry3d& body_from_camera) {
    constexpr uint64_t kMaxGapNs = 10'000'000;

    const auto after = std::ranges::lower_bound(samples, timestamp, {},
                                                &GroundTruthSample::timestamp);
    if (after == samples.end() || after == samples.begin()) {
        return std::nullopt;
    }
    const auto before = std::prev(after);
    if (timestamp - before->timestamp > kMaxGapNs
        || after->timestamp - timestamp > kMaxGapNs) {
        return std::nullopt;
    }

    const double alpha
        = static_cast<double>(timestamp - before->timestamp)
          / static_cast<double>(after->timestamp - before->timestamp);

    Eigen::Isometry3d world_from_body = Eigen::Isometry3d::Identity();
    world_from_body.linear()
        = before->orientation.slerp(alpha, after->orientation)
              .toRotationMatrix();
    world_from_body.translation()
        = (1.0 - alpha) * before->position + alpha * after->position;

    return world_from_body * body_from_camera;
}

Eigen::Isometry3d BodyFromCamera(const data::CamSensorParams& camera) {
    // T_BS from the sensor YAML, re-orthonormalised (it is printed to 1e-8)
    Eigen::Isometry3d body_from_camera = Eigen::Isometry3d::Identity();
    body_from_camera.linear()
        = Eigen::Quaterniond(Eigen::Matrix3d(camera.T_BS.topLeftCorner<3, 3>()))
              .normalized()
              .toRotationMatrix();
    body_from_camera.translation() = camera.T_BS.topRightCorner<3, 1>();
    return body_from_camera;
}

double RotationAngleDeg(const Eigen::Matrix3d& rotation) {
    return map::util::ComputeRotationAngle(rotation) * kRadToDeg;
}

double DirectionAngleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0))
           * kRadToDeg;
}

// Ground truth for p_curr = R * p_prev + t between two camera poses
struct RelativeTruth {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
};

struct PairScore {
    std::optional<RelativeTruth> truth;
    std::optional<double> v1_rotation_error_deg;
    std::optional<double> v2_rotation_error_deg;
    std::optional<double> v1_direction_error_deg;
    std::optional<double> v2_direction_error_deg;
    std::optional<double> v1_v2_rotation_deg;
    std::optional<double> v1_v2_direction_deg;
};

struct Summary {
    size_t count = 0;
    double median = 0.0;
    double p90 = 0.0;
    double max = 0.0;
};

Summary Summarise(std::vector<double> values) {
    Summary summary{.count = values.size()};
    if (values.empty()) {
        return summary;
    }
    std::ranges::sort(values);
    const auto at = [&](double quantile) {
        const auto index = static_cast<size_t>(
            std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
        return values.at(std::min(index, values.size() - 1));
    };
    summary.median = values.size() % 2 == 1
                         ? values[values.size() / 2]
                         : 0.5
                               * (values[values.size() / 2 - 1]
                                  + values[values.size() / 2]);
    summary.p90 = at(0.9);
    summary.max = values.back();
    return summary;
}

std::string FormatOptional(const std::optional<double>& value) {
    return value ? std::format("{:.3f}", *value) : std::string("-");
}

std::string FormatSummary(std::string_view label, const Summary& summary) {
    return std::format("| {} | {} | {:.3f} | {:.3f} | {:.3f} |\n", label,
                       summary.count, summary.median, summary.p90, summary.max);
}

std::vector<double> Collect(const std::vector<PairScore>& scores,
                            std::optional<double> PairScore::* field) {
    std::vector<double> values;
    for (const PairScore& score : scores) {
        if (const auto& value = score.*field) {
            values.push_back(*value);
        }
    }
    return values;
}

const char* EnvOrNull(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

}  // namespace

TEST(PoseEstimateDatasetTest, V2TracksGroundTruthAndV1OnV101Keyframes) {
    const char* euroc_dir = EnvOrNull("EUROC_DIR");
    if (euroc_dir == nullptr || EnvOrNull("WSLAM_SHADER_SRC_DIR") == nullptr
        || EnvOrNull("WSLAM_ONNX_MODEL") == nullptr) {
        GTEST_SKIP() << "needs EUROC_DIR, WSLAM_SHADER_SRC_DIR and "
                        "WSLAM_ONNX_MODEL (source .env)";
    }

    // The pipelines log every pass; keep the output to the report unless
    // SPDLOG_LEVEL asks otherwise
    spdlog::set_level(spdlog::level::warn);
    spdlog::cfg::load_env_levels();

    const auto schedule = LoadKeyframeSchedule(kSchedulePath);
    ASSERT_TRUE(schedule.has_value()) << schedule.error();

    const auto ground_truth
        = LoadGroundTruth(std::filesystem::path(euroc_dir)
                          / "state_groundtruth_estimate0" / "data.csv");
    ASSERT_TRUE(ground_truth.has_value()) << ground_truth.error();

    const auto v1 = RunV1(*schedule);
    ASSERT_TRUE(v1.has_value()) << "v1: " << v1.error();
    const auto v2 = RunV2(*schedule);
    ASSERT_TRUE(v2.has_value()) << "v2: " << v2.error();

    const size_t pair_count = schedule->frame_indices.size() - 1;
    ASSERT_EQ(v1->pairs.size(), pair_count);
    ASSERT_EQ(v2->pairs.size(), pair_count);

    const Eigen::Isometry3d body_from_camera = BodyFromCamera(v2->camera);

    std::vector<PairScore> scores(pair_count);
    std::string rows;
    size_t v1_recovered = 0;
    size_t v2_recovered = 0;

    for (size_t i = 0; i < pair_count; ++i) {
        const PairEstimate& e1 = v1->pairs[i];
        const PairEstimate& e2 = v2->pairs[i];
        // Both runs must have estimated the same keyframe pair
        ASSERT_EQ(e1.prev_ts, schedule->timestamps_ns[i]) << "pair " << i;
        ASSERT_EQ(e1.curr_ts, schedule->timestamps_ns[i + 1]) << "pair " << i;
        ASSERT_EQ(e2.prev_ts, e1.prev_ts) << "pair " << i;
        ASSERT_EQ(e2.curr_ts, e1.curr_ts) << "pair " << i;

        v1_recovered += e1.recovered ? 1 : 0;
        v2_recovered += e2.recovered ? 1 : 0;

        PairScore& score = scores[i];
        const auto prev_pose
            = CameraPoseAt(*ground_truth, e1.prev_ts, body_from_camera);
        const auto curr_pose
            = CameraPoseAt(*ground_truth, e1.curr_ts, body_from_camera);
        if (prev_pose && curr_pose) {
            const Eigen::Isometry3d curr_from_prev
                = curr_pose->inverse() * *prev_pose;
            score.truth = RelativeTruth{
                .rotation = curr_from_prev.linear(),
                .translation = curr_from_prev.translation(),
            };
        }

        const bool score_direction
            = score.truth
              && score.truth->translation.norm() >= kMinScoredBaselineM;
        const auto rotation_error = [&](const PairEstimate& estimate) {
            return RotationAngleDeg(estimate.rotation
                                    * score.truth->rotation.transpose());
        };
        const auto direction_error = [&](const PairEstimate& estimate) {
            return DirectionAngleDeg(estimate.translation,
                                     score.truth->translation);
        };

        if (score.truth && e1.recovered) {
            score.v1_rotation_error_deg = rotation_error(e1);
            if (score_direction) {
                score.v1_direction_error_deg = direction_error(e1);
            }
        }
        if (score.truth && e2.recovered) {
            score.v2_rotation_error_deg = rotation_error(e2);
            if (score_direction) {
                score.v2_direction_error_deg = direction_error(e2);
            }
        }
        if (e1.recovered && e2.recovered) {
            score.v1_v2_rotation_deg
                = RotationAngleDeg(e1.rotation * e2.rotation.transpose());
            score.v1_v2_direction_deg
                = DirectionAngleDeg(e1.translation, e2.translation);
        }

        rows += std::format(
            "| {} | {}-{} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | "
            "{}/{} | {} | {} | {:.3f} | {:.3f} |\n",
            i, schedule->frame_indices[i], schedule->frame_indices[i + 1],
            FormatOptional(score.truth ? std::optional(RotationAngleDeg(
                                             score.truth->rotation))
                                       : std::nullopt),
            FormatOptional(score.truth
                               ? std::optional(score.truth->translation.norm())
                               : std::nullopt),
            FormatOptional(score.v1_rotation_error_deg),
            FormatOptional(score.v2_rotation_error_deg),
            FormatOptional(score.v1_direction_error_deg),
            FormatOptional(score.v2_direction_error_deg),
            FormatOptional(score.v1_v2_rotation_deg),
            FormatOptional(score.v1_v2_direction_deg), e1.inliers, e1.matches,
            e2.inliers, e2.matches, e1.landmarks, e2.landmarks,
            e1.mean_reprojection_px, e2.mean_reprojection_px);
    }

    const Summary v1_rotation
        = Summarise(Collect(scores, &PairScore::v1_rotation_error_deg));
    const Summary v2_rotation
        = Summarise(Collect(scores, &PairScore::v2_rotation_error_deg));
    const Summary v1_direction
        = Summarise(Collect(scores, &PairScore::v1_direction_error_deg));
    const Summary v2_direction
        = Summarise(Collect(scores, &PairScore::v2_direction_error_deg));
    const Summary v1_v2_rotation
        = Summarise(Collect(scores, &PairScore::v1_v2_rotation_deg));
    const Summary v1_v2_direction
        = Summarise(Collect(scores, &PairScore::v1_v2_direction_deg));

    std::string report = std::format(
        "## Pose estimate v1 vs v2 on {} ({} keyframes, {} pairs)\n\n"
        "Recovered poses: v1 {}/{}, v2 {}/{}. Wall time incl. setup: v1 "
        "{:.1f} s, v2 {:.1f} s.\n"
        "Direction errors only for GT baselines >= {} m.\n\n"
        "| metric (deg) | n | median | p90 | max |\n|---|---|---|---|---|\n",
        schedule->sequence, schedule->frame_indices.size(), pair_count,
        v1_recovered, pair_count, v2_recovered, pair_count, v1->seconds,
        v2->seconds, kMinScoredBaselineM);
    report += FormatSummary("v1 rotation error vs GT", v1_rotation);
    report += FormatSummary("v2 rotation error vs GT", v2_rotation);
    report
        += FormatSummary("v1 translation direction error vs GT", v1_direction);
    report
        += FormatSummary("v2 translation direction error vs GT", v2_direction);
    report += FormatSummary("v1 vs v2 rotation", v1_v2_rotation);
    report += FormatSummary("v1 vs v2 translation direction", v1_v2_direction);
    report += "\n| # | frames | GT rot | GT baseline (m) | v1 rot err | v2 rot "
              "err | v1 dir err | v2 dir err | v1-v2 rot | v1-v2 dir | v1 "
              "inl/matches | v2 inl/matches | v1 lm | v2 lm | v1 reproj px | "
              "v2 reproj px |\n";
    report += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
              "\n";
    report += rows;

    std::println("{}", report);
    if (const char* out = EnvOrNull("WSLAM_POSE_REPORT_OUT")) {
        std::ofstream(out) << report;
    }

    // Thresholds fixed before the first run. v1 is the guideline, the
    // ground truth the referee
    EXPECT_GE(static_cast<double>(v2_recovered),
              0.9 * static_cast<double>(pair_count));
    EXPECT_LE(v2_rotation.median, 0.5);
    EXPECT_LE(v2_direction.median, 10.0);
    EXPECT_LE(v2_rotation.median,
              std::max(1.5 * v1_rotation.median, v1_rotation.median + 0.1))
        << "v2 should not be clearly worse than v1";
    EXPECT_LE(v1_v2_rotation.median, 1.0);
}
