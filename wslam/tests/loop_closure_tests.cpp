// Unit tests of the v2 loop closure building blocks that need no model or
// dataset: the model input/output adapters, the FAISS place database, PnP
// verification and the pose-only graph

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "feature_detect_v2/onnx_utils.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"
#include "map_v2/loop_closure/config.hpp"
#include "map_v2/loop_closure/model_io.hpp"
#include "map_v2/loop_closure/place_database.hpp"
#include "map_v2/loop_closure/pnp_verify.hpp"
#include "map_v2/loop_closure/pose_graph.hpp"

namespace lc = wslam::v2::mapping::loop_closure;
namespace model = wslam::v2::raco_lightglue;

namespace {
constexpr uint16_t kWidth = 752;
constexpr uint16_t kHeight = 480;

lc::PlaceImage MakeImage(uint16_t width, uint16_t height,
                         const auto& value_at) {
    lc::PlaceImage image{.width = width, .height = height, .pixels = {}};
    image.pixels.resize(size_t{width} * height);
    for (uint16_t y = 0; y < height; ++y) {
        for (uint16_t x = 0; x < width; ++x) {
            image.pixels[size_t{y} * width + x]
                = static_cast<uint8_t>(value_at(x, y));
        }
    }
    return image;
}

// Undoes ToVprInput's ImageNet normalisation of channel c
float Denormalise(float value, size_t channel) {
    constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
    constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
    return (value * kStd[channel] + kMean[channel]) * 255.0F;
}

lc::PlaceDescriptor UnitDescriptor(std::vector<float> values) {
    return lc::PlaceDescriptor::FromVprOutput(values, values.size()).value();
}
}  // namespace

// ---------------------------------------------------------------- model IO

TEST(LoopClosureModelIoTest, FrameRoundTripsToEightBits) {
    wslam::data::FrameBW frame{.timestamp = 1,
                               .pixels = {0.0F, 1.0F / 255.0F, 0.5F, 1.0F},
                               .width = 2,
                               .height = 2};
    const lc::PlaceImage image = lc::PlaceImage::FromFrame(frame);
    EXPECT_EQ(image.width, 2);
    EXPECT_EQ(image.height, 2);
    EXPECT_EQ(image.pixels, (std::vector<uint8_t>{0, 1, 128, 255}));
}

TEST(LoopClosureModelIoTest, VprInputIsNormalisedNchw) {
    const auto image = MakeImage(kWidth, kHeight, [](int, int) { return 128; });
    const lc::VprInputSpec spec{.width = 320, .height = 320};
    const std::vector<float> input = lc::ToVprInput(image, spec);

    ASSERT_EQ(input.size(), spec.elementCount());
    const size_t plane = size_t{spec.width} * spec.height;
    for (size_t channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(Denormalise(input[channel * plane], channel), 128.0F,
                    1e-3F);
        EXPECT_NEAR(Denormalise(input[channel * plane + plane - 1], channel),
                    128.0F, 1e-3F);
    }
}

TEST(LoopClosureModelIoTest, SquashAveragesTheCoveredArea) {
    // Left half black, right half white
    const auto image = MakeImage(
        kWidth, kHeight, [](int x, int) { return x < kWidth / 2 ? 0 : 255; });

    // Two columns: each covers exactly one half
    const auto halves = lc::ToVprInput(image, {.width = 2, .height = 4});
    EXPECT_NEAR(Denormalise(halves[0], 0), 0.0F, 1e-3F);
    EXPECT_NEAR(Denormalise(halves[1], 0), 255.0F, 1e-3F);

    // Three columns: the middle one straddles the edge evenly
    const auto thirds = lc::ToVprInput(image, {.width = 3, .height = 4});
    EXPECT_NEAR(Denormalise(thirds[0], 0), 0.0F, 1e-3F);
    EXPECT_NEAR(Denormalise(thirds[1], 0), 127.5F, 1e-2F);
    EXPECT_NEAR(Denormalise(thirds[2], 0), 255.0F, 1e-3F);
}

TEST(LoopClosureModelIoTest, MatcherInputIsCentreCroppedImage) {
    const auto image = MakeImage(kWidth, kHeight,
                                 [](int x, int y) { return (x + y) % 256; });
    const std::vector<float> input = lc::ToMatcherInput(image);
    ASSERT_EQ(input.size(), model::kImagesBytes / sizeof(float));

    const size_t plane = size_t{model::kInputWidth} * model::kInputHeight;
    const auto at = [&](size_t channel, uint32_t x, uint32_t y) {
        return input[channel * plane + size_t{y} * model::kInputWidth + x];
    };
    for (const auto [x, y] :
         {std::pair{0U, 0U}, std::pair{100U, 200U},
          std::pair{model::kInputWidth - 1, model::kInputHeight - 1}}) {
        const uint32_t sx = x + model::kCropX;
        const uint32_t sy = y + model::kCropY;
        for (size_t channel = 0; channel < 3; ++channel) {
            EXPECT_FLOAT_EQ(at(channel, x, y),
                            static_cast<float>((sx + sy) % 256) / 255.0F);
        }
    }
}

TEST(LoopClosureModelIoTest, DescriptorIsRenormalised) {
    const auto descriptor
        = lc::PlaceDescriptor::FromVprOutput(std::vector{3.0F, 4.0F}, 2);
    ASSERT_TRUE(descriptor.has_value());
    EXPECT_NEAR(descriptor->values[0], 0.6F, 1e-6F);
    EXPECT_NEAR(descriptor->values[1], 0.8F, 1e-6F);
    EXPECT_NEAR(descriptor->similarity(*descriptor), 1.0F, 1e-6F);

    EXPECT_FALSE(
        lc::PlaceDescriptor::FromVprOutput(std::vector{1.0F}, 2).has_value());
    EXPECT_FALSE(lc::PlaceDescriptor::FromVprOutput(std::vector{0.0F, 0.0F}, 2)
                     .has_value());
}

TEST(LoopClosureModelIoTest, MatcherOutputToSourcePixels) {
    constexpr int64_t kKeypoints = 2;
    const std::vector<model::KeypointPosition> left{
        {.x = 10.5F, .y = 20.5F},
        {.x = 1.5F, .y = 2.5F},
    };
    const std::vector<model::KeypointPosition> right{
        {.x = 30.5F, .y = 40.5F},
        {.x = 3.5F, .y = 4.5F},
    };
    // Zero-padded to kKeypoints rows: the second row is padding
    const std::vector<model::KeypointMatch> matches{
        {.pair_index = 0, .left_index = 1, .right_index = 0},
        {.pair_index = 0, .left_index = 0, .right_index = 0},
    };
    const std::vector<float> scores{0.9F, 0.0F};

    const auto pairs = lc::PairMatches::FromMatcherOutput(left, right, matches,
                                                          scores, kKeypoints);
    ASSERT_TRUE(pairs.has_value());
    ASSERT_EQ(pairs->matches.size(), 1U);
    const lc::PairMatch& match = pairs->matches.front();
    EXPECT_EQ(match.left_index, 1U);
    EXPECT_EQ(match.right_index, 0U);
    EXPECT_FLOAT_EQ(match.score, 0.9F);
    EXPECT_DOUBLE_EQ(match.left_px.x(), 1.0 + model::kCropX);
    EXPECT_DOUBLE_EQ(match.left_px.y(), 2.0 + model::kCropY);
    EXPECT_DOUBLE_EQ(match.right_px.x(), 30.0 + model::kCropX);

    const std::vector<model::KeypointMatch> malformed{
        {.pair_index = 0, .left_index = 2, .right_index = 0},
        {.pair_index = 0, .left_index = 0, .right_index = 0},
    };
    EXPECT_FALSE(lc::PairMatches::FromMatcherOutput(left, right, malformed,
                                                    scores, kKeypoints)
                     .has_value());
}

TEST(LoopClosureModelIoTest, ParseMatchesDropsPadding) {
    constexpr int64_t kKeypoints = 4;
    const std::vector<model::KeypointMatch> padded{
        {.pair_index = 0, .left_index = 3, .right_index = 1},
        {.pair_index = 0, .left_index = 0, .right_index = 2},
        {.pair_index = 0, .left_index = 0, .right_index = 0},
        {.pair_index = 0, .left_index = 0, .right_index = 0},
    };
    const std::vector<float> scores{0.7F, 0.2F, 0.0F, 0.0F};

    const auto parsed = wslam::v2::ParseMatches(padded, scores, kKeypoints);
    ASSERT_TRUE(parsed.has_value());
    ASSERT_EQ(parsed->matches.size(), 2U);
    EXPECT_EQ(parsed->matches[0].left_index, 3);
    EXPECT_EQ(parsed->matches[1].right_index, 2);
    EXPECT_FLOAT_EQ(parsed->scores[1], 0.2F);

    // A padded row with a score is a malformed match, not padding
    const std::vector<float> scored_padding{0.7F, 0.2F, 0.1F, 0.0F};
    const std::vector<model::KeypointMatch> out_of_range{
        {.pair_index = 0, .left_index = 4, .right_index = 1},
        {.pair_index = 0, .left_index = 0, .right_index = 2},
        {.pair_index = 0, .left_index = 0, .right_index = 0},
        {.pair_index = 0, .left_index = 0, .right_index = 0},
    };
    EXPECT_FALSE(
        wslam::v2::ParseMatches(out_of_range, scored_padding, kKeypoints)
            .has_value());
    EXPECT_FALSE(
        wslam::v2::ParseMatches(padded, std::vector<float>{0.5F}, kKeypoints)
            .has_value());
}

// ---------------------------------------------------------------- database

TEST(PlaceDatabaseTest, ReturnsMostSimilarWithinIdRange) {
    lc::PlaceDatabase database(3);
    database.add(1, UnitDescriptor({1.0F, 0.0F, 0.0F}));
    database.add(2, UnitDescriptor({0.0F, 1.0F, 0.0F}));
    database.add(5, UnitDescriptor({0.9F, 0.1F, 0.0F}));
    database.add(9, UnitDescriptor({1.0F, 0.0F, 0.0F}));
    EXPECT_EQ(database.size(), 4U);

    const auto query = UnitDescriptor({1.0F, 0.0F, 0.0F});

    // Keyframe 9 is too recent
    const auto candidates = database.search(query, 2, 5);
    ASSERT_EQ(candidates.size(), 2U);
    EXPECT_EQ(candidates[0].keyframe, 1U);
    EXPECT_NEAR(candidates[0].similarity, 1.0F, 1e-6F);
    EXPECT_EQ(candidates[1].keyframe, 5U);

    // Fewer allowed than asked for
    const auto only_first = database.search(query, 3, 1);
    ASSERT_EQ(only_first.size(), 1U);
    EXPECT_EQ(only_first[0].keyframe, 1U);

    EXPECT_TRUE(database.search(query, 3, 0).empty());
}

// --------------------------------------------------------------------- PnP

namespace {
struct PnpScene {
    std::vector<lc::PnpCorrespondence> correspondences;
    gtsam::Pose3 T_candidate_query;
};

// Points in front of both cameras; a fraction of the rays replaced by noise
PnpScene MakePnpScene(size_t count, double outlier_fraction, double noise_px,
                      double focal) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> lateral(-3.0, 3.0);
    std::uniform_real_distribution<double> depth(3.0, 9.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> ray(-0.6, 0.6);
    std::normal_distribution<double> noise(0.0, noise_px / focal);

    PnpScene scene;
    scene.T_candidate_query = gtsam::Pose3{gtsam::Rot3::RzRyRx(0.05, -0.2, 0.1),
                                           gtsam::Point3{0.6, -0.1, 0.3}};
    const gtsam::Pose3 query_from_candidate = scene.T_candidate_query.inverse();

    while (scene.correspondences.size() < count) {
        const Eigen::Vector3d point{lateral(rng), lateral(rng), depth(rng)};
        const Eigen::Vector3d in_query
            = query_from_candidate.transformFrom(gtsam::Point3{point});
        if (in_query.z() < 1.0) {
            continue;
        }
        Eigen::Vector2d observed = in_query.head<2>() / in_query.z();
        if (unit(rng) < outlier_fraction) {
            observed = {ray(rng), ray(rng)};
        } else {
            observed += Eigen::Vector2d{noise(rng), noise(rng)};
        }
        scene.correspondences.push_back(
            {.point_candidate = point, .query_ray = observed});
    }
    return scene;
}
}  // namespace

TEST(PnpVerifyTest, RecoversPoseWithOutliers) {
    constexpr double kFocal = 458.0;
    const PnpScene scene = MakePnpScene(120, 0.4, 0.5, kFocal);

    const lc::MatchAcceptance acceptance{};
    const auto result = lc::SolvePnp(scene.correspondences, kFocal, acceptance);
    ASSERT_TRUE(result.has_value()) << result.error();

    const gtsam::Pose3 error
        = scene.T_candidate_query.between(result->T_candidate_query);
    EXPECT_LT(error.translation().norm(), 0.02);
    EXPECT_LT(error.rotation().axisAngle().second, 0.002);
    EXPECT_GE(result->inliers, 60U);
    EXPECT_LE(result->inliers, 85U);
    EXPECT_EQ(result->correspondences, 120U);
}

TEST(PnpVerifyTest, RejectsTooFewCorrespondencesOrInliers) {
    constexpr double kFocal = 458.0;
    const lc::MatchAcceptance acceptance{};

    const PnpScene few = MakePnpScene(10, 0.0, 0.5, kFocal);
    EXPECT_FALSE(
        lc::SolvePnp(few.correspondences, kFocal, acceptance).has_value());

    // Mostly noise: whatever RANSAC finds misses the inlier gates
    const PnpScene noise = MakePnpScene(100, 0.95, 0.5, kFocal);
    EXPECT_FALSE(
        lc::SolvePnp(noise.correspondences, kFocal, acceptance).has_value());
}

// -------------------------------------------------------------- pose graph

namespace {
// Camera frame world (as the VIO's): y down, so up is -y
const Eigen::Vector3d kGravity{0.0, 9.81, 0.0};
const Eigen::Vector3d kUp{0.0, -1.0, 0.0};

// A camera driving a circle of radius 2 m about the up axis, looking along
// its direction of travel
gtsam::Pose3 CirclePose(double angle) {
    const gtsam::Rot3 yaw = gtsam::Rot3::AxisAngle(gtsam::Unit3{kUp}, angle);
    const Eigen::Vector3d position{2.0 * std::cos(angle), 0.0,
                                   2.0 * std::sin(angle)};
    return gtsam::Pose3{yaw, gtsam::Point3{position}};
}
}  // namespace

TEST(PoseGraphTest, ProjectToYawKeepsOnlyTheTwist) {
    const gtsam::Rot3 yaw = gtsam::Rot3::AxisAngle(gtsam::Unit3{kUp}, 0.3);
    const gtsam::Rot3 roll = gtsam::Rot3::Rz(0.1);
    const gtsam::Pose3 drift{yaw * roll, gtsam::Point3{1.0, 2.0, 3.0}};

    const gtsam::Pose3 projected = lc::ProjectToYaw(drift, kUp);
    const auto [axis, angle] = projected.rotation().axisAngle();
    EXPECT_NEAR(std::abs(axis.point3().dot(kUp)), 1.0, 1e-9);
    EXPECT_NEAR(angle, 0.3, 0.01);
    EXPECT_TRUE(projected.translation().isApprox(drift.translation()));
}

TEST(PoseGraphTest, NodeGateAttachesNearDuplicates) {
    lc::PoseGraph graph(lc::PoseGraphNoise{}, lc::NodeGate{});
    graph.setGravity(kGravity);

    const gtsam::Pose3 first = CirclePose(0.0);
    EXPECT_TRUE(graph.addKeyframe(1, first));
    // 1 cm and no rotation away: attached to node 1
    EXPECT_FALSE(graph.addKeyframe(
        2,
        first.compose(gtsam::Pose3{gtsam::Rot3{}, gtsam::Point3{0.01, 0, 0}})));
    EXPECT_TRUE(graph.addKeyframe(3, CirclePose(0.2)));

    EXPECT_TRUE(graph.isNode(1));
    EXPECT_FALSE(graph.isNode(2));
    EXPECT_TRUE(graph.isNode(3));
    EXPECT_EQ(graph.nodeCount(), 2U);
    EXPECT_EQ(graph.keyframeCount(), 3U);

    // Without loops, every correction is the identity
    EXPECT_TRUE(graph.correctionFor(2).equals(gtsam::Pose3{}, 1e-6));
    EXPECT_TRUE(graph.correctedPose(3).equals(CirclePose(0.2), 1e-6));
}

TEST(PoseGraphTest, AttachedKeyframesBlendTheirNodesCorrections) {
    lc::PoseGraph graph(lc::PoseGraphNoise{}, lc::NodeGate{});
    graph.setGravity(kGravity);

    // Nodes every 0.3 m along x, each followed by two attached keyframes
    // 4 and 8 cm on (under the gate's 10 cm); the VIO drifts 2 cm per node in
    // y
    const gtsam::Rot3 level{};
    uint64_t keyframe = 1;
    std::vector<uint64_t> nodes;
    for (int n = 0; n < 6; ++n) {
        const double x = 0.3 * n;
        const double drift = 0.02 * n;
        ASSERT_TRUE(graph.addKeyframe(
            keyframe, gtsam::Pose3{level, gtsam::Point3{x, drift, 0.0}}));
        nodes.push_back(keyframe++);
        for (const double offset : {0.04, 0.08}) {
            ASSERT_FALSE(graph.addKeyframe(
                keyframe++,
                gtsam::Pose3{level, gtsam::Point3{x + offset, drift, 0.0}}));
        }
    }

    // The truth is drift-free: a loop from the first node to the last
    graph.addLoop(lc::LoopEdge{
        .candidate = nodes.front(),
        .query = nodes.back(),
        .T_candidate_query = gtsam::Pose3{level, gtsam::Point3{1.5, 0, 0}},
    });

    // Second node's attached keyframes, by VIO path length: 4 and 8 cm of
    // 8 cm plus the last leg to the third node (0.22 m on, 0.02 m across)
    const uint64_t node = nodes[1];
    const gtsam::Pose3 from = graph.correctionFor(node);
    const gtsam::Pose3 to = graph.correctionFor(nodes[2]);
    ASSERT_GT((from.translation() - to.translation()).norm(), 1e-3);

    EXPECT_TRUE(graph.smoothCorrectionFor(node).equals(from, 1e-9));
    const double segment = 0.08 + std::hypot(0.22, 0.02);
    EXPECT_TRUE(graph.smoothCorrectionFor(node + 1)
                    .equals(from.interpolateRt(to, 0.04 / segment), 1e-6));
    EXPECT_TRUE(graph.smoothCorrectionFor(node + 2)
                    .equals(from.interpolateRt(to, 0.08 / segment), 1e-6));
    // The step correction holds the node's for both
    EXPECT_TRUE(graph.correctionFor(node + 2).equals(from, 1e-9));

    // After the last node: the last node's correction
    const uint64_t last = keyframe - 1;
    EXPECT_TRUE(graph.smoothCorrectionFor(last).equals(
        graph.correctionFor(nodes.back()), 1e-9));
}

TEST(PoseGraphTest, LoopRemovesYawAndPositionDrift) {
    lc::PoseGraph graph(lc::PoseGraphNoise{}, lc::NodeGate{});
    graph.setGravity(kGravity);

    // VIO drifts in yaw (0.5 deg per keyframe) and position (1 cm per
    // keyframe); roll and pitch stay right (gravity observes them)
    constexpr int kKeyframes = 40;
    const double step = 2.0 * M_PI / kKeyframes;
    const gtsam::Rot3 yaw_drift_step
        = gtsam::Rot3::AxisAngle(gtsam::Unit3{kUp}, 0.5 * M_PI / 180.0);

    std::vector<gtsam::Pose3> truth;
    gtsam::Pose3 vio = CirclePose(0.0);
    for (int k = 0; k <= kKeyframes; ++k) {
        truth.push_back(CirclePose(step * k));
        if (k > 0) {
            const gtsam::Pose3 true_step = truth[k - 1].between(truth[k]);
            const gtsam::Pose3 drifted_step{
                yaw_drift_step * true_step.rotation(),
                true_step.translation() + gtsam::Point3{0.01, 0.0, 0.0}};
            vio = vio.compose(drifted_step);
        }
        ASSERT_TRUE(graph.addKeyframe(static_cast<uint64_t>(k + 1), vio));
    }

    const uint64_t last = kKeyframes + 1;
    const double drift_before
        = (graph.vioPose(last).translation() - truth.back().translation())
              .norm();
    ASSERT_GT(drift_before, 0.3);

    // The last keyframe is back at the start: PnP measured the truth
    graph.addLoop(lc::LoopEdge{
        .candidate = 1,
        .query = last,
        .T_candidate_query = truth.front().between(truth.back()),
    });

    const double drift_after
        = (graph.correctedPose(last).translation() - truth.back().translation())
              .norm();
    EXPECT_LT(drift_after, 0.1 * drift_before);

    const auto rotation_error = [&](const gtsam::Pose3& pose) {
        return pose.rotation()
            .between(truth.back().rotation())
            .axisAngle()
            .second;
    };
    EXPECT_LT(rotation_error(graph.correctedPose(last)),
              0.1 * rotation_error(graph.vioPose(last)));

    // The drift handed to the main loop rotates about gravity only
    const auto [axis, angle] = graph.driftCorrection().rotation().axisAngle();
    if (angle > 1e-6) {
        EXPECT_NEAR(std::abs(axis.point3().dot(kUp)), 1.0, 1e-6);
    }
    EXPECT_EQ(graph.loopCount(), 1U);
}
