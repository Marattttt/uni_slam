// The frame prefilter's IMU motion prediction (feature_detect_v2/
// motion_prediction.hpp) and pass, on synthetic IMU data. No dataset needed

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "common.hpp"
#include "compute/stage.hpp"
#include "data/provider_base.hpp"
#include "feature_detect_v2/frame_prefilter.hpp"
#include "feature_detect_v2/motion_prediction.hpp"
#include "map/utils.hpp"
#include "models.hpp"

namespace {

using namespace wslam;

constexpr double kGravity = 9.81;
constexpr uint64_t kKeyframeTs = 1'000'000'000;
constexpr uint64_t kImuPeriodNs = 5'000'000;  // 200 Hz

// EuRoC cam0 intrinsics without distortion; camera = body = IMU frame
data::CamSensorParams Camera() {
    return {
        .T_BS = Eigen::Matrix4d::Identity(),
        .rate_hz = 20,
        .resolution = {752, 480},
        .camera_model = "pinhole",
        .intrinsics = {458.654, 457.296, 367.215, 248.375},
        .distortion_model = data::DistortionModel::RadialTangential,
        .distortion_coefficients = Eigen::Vector4d::Zero(),
    };
}

data::IMUSensorParams Imu() {
    return {
        .T_BS = Eigen::Matrix4d::Identity(),
        .rate_hz = 200,
        .gyroscope_noise_density = 1.6968e-04,
        .gyroscope_random_walk = 1.9393e-05,
        .accelerometer_noise_density = 2.0e-3,
        .accelerometer_random_walk = 3.0e-3,
    };
}

// Keyframe at rest orientation; gravity along world -z
v2::KeyframeMotionPrior Prior(const Eigen::Vector3d& velocity, double depth) {
    return {
        .keyframe_ts = kKeyframeTs,
        .R_world_cam = Eigen::Matrix3d::Identity(),
        .velocity_world = velocity,
        .gravity_world = {0.0, 0.0, -kGravity},
        .accel_bias = Eigen::Vector3d::Zero(),
        .gyro_bias = Eigen::Vector3d::Zero(),
        .median_depth_m = depth,
        .velocity_sigma = 0.0,
        .min_rotation_rad = 0.05,
        .min_parallax_px = 20.0,
    };
}

// Samples after the keyframe up to duration_s, with a constant angular
// velocity and the specific force of an unaccelerated body
std::vector<data::IMUReading> Samples(double duration_s,
                                      const Eigen::Vector3d& gyro) {
    std::vector<data::IMUReading> samples;
    const auto count = static_cast<uint64_t>(duration_s * 1e9) / kImuPeriodNs;
    for (uint64_t i = 1; i <= count; ++i) {
        data::IMUReading reading{.timestamp = kKeyframeTs + i * kImuPeriodNs,
                                 .vals = {}};
        reading.vals << static_cast<float>(gyro.x()),
            static_cast<float>(gyro.y()), static_cast<float>(gyro.z()), 0.0F,
            0.0F, static_cast<float>(kGravity);
        samples.push_back(reading);
    }
    return samples;
}

uint64_t FrameTs(double seconds) {
    return kKeyframeTs + static_cast<uint64_t>(seconds * 1e9);
}

v2::KeyframeMotionPrediction Predict(const v2::KeyframeMotionPrior& prior,
                                     double duration_s,
                                     const Eigen::Vector3d& gyro) {
    v2::MotionPredictor predictor{Camera(), Imu(),
                                  v2::MotionPredictionOptions{}};
    const auto prediction = predictor.predict(prior, Samples(duration_s, gyro),
                                              FrameTs(duration_s));
    EXPECT_TRUE(prediction.has_value()) << prediction.error();
    return prediction.value_or(v2::KeyframeMotionPrediction{});
}

TEST(MotionPredictionTest, StationaryIsSmallMotion) {
    const auto prediction = Predict(Prior(Eigen::Vector3d::Zero(), 4.0), 0.2,
                                    Eigen::Vector3d::Zero());

    EXPECT_NEAR(prediction.dt_s, 0.2, 1e-9);
    EXPECT_NEAR(prediction.rotation_rad, 0.0, 1e-6);
    EXPECT_NEAR(prediction.parallax_px, 0.0, 1e-3);
    EXPECT_GT(prediction.rotation_upper_rad, prediction.rotation_rad);
    EXPECT_GT(prediction.parallax_upper_px, prediction.parallax_px);
    EXPECT_TRUE(prediction.small_motion);
}

TEST(MotionPredictionTest, ConstantTurnRateGivesItsAngle) {
    const auto prediction = Predict(Prior(Eigen::Vector3d::Zero(), 4.0), 0.2,
                                    Eigen::Vector3d{0.0, 0.0, 0.5});

    EXPECT_NEAR(prediction.rotation_rad, 0.1, 1e-4);
    EXPECT_FALSE(prediction.small_motion);
}

TEST(MotionPredictionTest, LateralMotionParallaxScalesWithDepth) {
    // 1 m/s along the camera x axis for 0.1 s past a scene 5 m away: every
    // point shifts by f * 0.1 / 5 pixels
    const auto prediction = Predict(Prior(Eigen::Vector3d{1.0, 0.0, 0.0}, 5.0),
                                    0.1, Eigen::Vector3d::Zero());
    EXPECT_NEAR(prediction.parallax_px, 458.654 * 0.1 / 5.0, 0.05);
    EXPECT_NEAR(prediction.rotation_rad, 0.0, 1e-6);
    EXPECT_GT(prediction.parallax_upper_px, prediction.parallax_px);

    const auto further = Predict(Prior(Eigen::Vector3d{1.0, 0.0, 0.0}, 5.0),
                                 0.3, Eigen::Vector3d::Zero());
    EXPECT_NEAR(further.parallax_px, 458.654 * 0.3 / 5.0, 0.1);
    EXPECT_FALSE(further.small_motion);
}

TEST(MotionPredictionTest, UpperBoundsGrowWithTime) {
    const auto prior = Prior(Eigen::Vector3d::Zero(), 4.0);
    const auto early = Predict(prior, 0.05, Eigen::Vector3d::Zero());
    const auto late = Predict(prior, 0.5, Eigen::Vector3d::Zero());
    EXPECT_GT(late.rotation_upper_rad, early.rotation_upper_rad);
    EXPECT_GT(late.parallax_upper_px, early.parallax_upper_px);
}

TEST(MotionPredictionTest, RejectsUnusablePriors) {
    v2::MotionPredictor predictor{Camera(), Imu(),
                                  v2::MotionPredictionOptions{}};
    const auto samples = Samples(0.1, Eigen::Vector3d::Zero());

    EXPECT_FALSE(predictor
                     .predict(Prior(Eigen::Vector3d::Zero(), 4.0), samples,
                              kKeyframeTs - 1)
                     .has_value());
    EXPECT_FALSE(
        predictor
            .predict(Prior(Eigen::Vector3d::Zero(), 0.0), samples, FrameTs(0.1))
            .has_value());
}

// ------------------------------------------------------------------- pass

class FramePrefilterPassTest : public ::testing::Test {
   protected:
    AnyBag storage;

    void SetUp() override {
        storage.set(ResourceIdentifier::GetCameraIntrinsicsName(0), Camera());
        storage.set(ResourceIdentifier::ImuParamsName, Imu());
        storage.set(ResourceIdentifier::GetImuVecName(),
                    Samples(1.0, Eigen::Vector3d::Zero()));
    }

    void setFrame(double seconds) {
        storage.set(ResourceIdentifier::FrameTimestampNsName, FrameTs(seconds));
    }
};

TEST_F(FramePrefilterPassTest, RunsWithoutPrior) {
    v2::FramePrefilterPass pass(storage, v2::FramePrefilterOptions{});
    ASSERT_EQ(pass.initialize(), std::nullopt);

    setFrame(0.1);
    EXPECT_EQ(pass.execute(), std::nullopt);
    EXPECT_FALSE(storage.has(ResourceIdentifier::FrameSkippedName));
}

TEST_F(FramePrefilterPassTest, SkipsSmallMotionUpToTheCap) {
    v2::FramePrefilterOptions options;
    options.max_consecutive_skips = 2;
    v2::FramePrefilterPass pass(storage, options);
    ASSERT_EQ(pass.initialize(), std::nullopt);
    storage.set(ResourceIdentifier::KeyframeMotionPriorName,
                Prior(Eigen::Vector3d::Zero(), 4.0));

    for (int frame = 1; frame <= 2; ++frame) {
        setFrame(0.05 * frame);
        EXPECT_EQ(pass.execute(), compute::kStageStopExecution);
        EXPECT_TRUE(storage.take<bool>(ResourceIdentifier::FrameSkippedName)
                        .value_or(false));
    }

    // The cap: the third small-motion frame in a row runs the model
    setFrame(0.15);
    EXPECT_EQ(pass.execute(), std::nullopt);
    EXPECT_FALSE(storage.has(ResourceIdentifier::FrameSkippedName));
    const auto prediction = storage.take<v2::KeyframeMotionPrediction>(
        ResourceIdentifier::KeyframeMotionPredictionName);
    ASSERT_TRUE(prediction.has_value());
    EXPECT_TRUE(prediction->small_motion);

    // And the count starts over
    setFrame(0.2);
    EXPECT_EQ(pass.execute(), compute::kStageStopExecution);
}

TEST_F(FramePrefilterPassTest, SkipsStationaryFramesBeforeAPrior) {
    v2::FramePrefilterOptions options;
    options.max_stationary_skips = 3;
    v2::FramePrefilterPass pass(storage, options);
    ASSERT_EQ(pass.initialize(), std::nullopt);

    // The first frame runs whatever the IMU says: the model needs a reference
    setFrame(0.5);
    EXPECT_EQ(pass.execute(), std::nullopt);

    // A parked IMU (constant readings): skipped up to the cap
    for (int frame = 1; frame <= 3; ++frame) {
        setFrame(0.5 + 0.05 * frame);
        EXPECT_EQ(pass.execute(), compute::kStageStopExecution);
        EXPECT_TRUE(storage.take<bool>(ResourceIdentifier::FrameSkippedName)
                        .value_or(false));
    }
    setFrame(0.7);
    EXPECT_EQ(pass.execute(), std::nullopt);
    EXPECT_FALSE(storage.has(ResourceIdentifier::FrameSkippedName));
}

TEST_F(FramePrefilterPassTest, RunsWhenTheImuMovesBeforeAPrior) {
    v2::FramePrefilterPass pass(storage, v2::FramePrefilterOptions{});
    ASSERT_EQ(pass.initialize(), std::nullopt);
    setFrame(0.5);
    ASSERT_EQ(pass.execute(), std::nullopt);

    // Shaking: the angular velocity alternates by 2 rad/s sample to sample
    auto samples = Samples(1.0, Eigen::Vector3d::Zero());
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i].vals[0] = (i % 2 == 0) ? 1.0F : -1.0F;
    }
    storage.set(ResourceIdentifier::GetImuVecName(), samples);

    setFrame(0.55);
    EXPECT_EQ(pass.execute(), std::nullopt);
    EXPECT_FALSE(storage.has(ResourceIdentifier::FrameSkippedName));
}

TEST(MotionPredictionTest, StationaryScoreSeparatesParkedFromShaking) {
    const auto parked = Samples(0.25, Eigen::Vector3d{0.08, 0.0, 0.0});
    EXPECT_NEAR(map::utils::StationaryScore(parked), 0.0, 1e-6);

    auto shaking = parked;
    for (size_t i = 0; i < shaking.size(); ++i) {
        shaking[i].vals[5] += (i % 2 == 0) ? 1.0F : -1.0F;
    }
    EXPECT_NEAR(map::utils::StationaryScore(shaking), 1.0, 1e-3);
}

TEST(MotionPredictionTest, PropagateFromRestFollowsTheGyroscope) {
    // 0.6 rad/s measured about x with a 0.1 rad/s bias: 0.5 rad in one second
    const auto samples = Samples(1.1, Eigen::Vector3d{0.6, 0.0, 0.0});
    const uint64_t from = kKeyframeTs;
    const uint64_t to = from + 1'000'000'000;

    const auto state = map::utils::PropagateFromRest(
        {0.0, 0.0, -kGravity}, samples, from, to,
        gtsam::imuBias::ConstantBias{Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d{0.1, 0.0, 0.0}},
        Camera(), Imu());

    // Seen from a camera turned by +0.5 rad about x: R_x(0.5)^T * g
    EXPECT_NEAR(state.gravity_cam.x(), 0.0, 1e-6);
    EXPECT_NEAR(state.gravity_cam.y(), -kGravity * std::sin(0.5), 1e-4);
    EXPECT_NEAR(state.gravity_cam.z(), -kGravity * std::cos(0.5), 1e-4);
}

TEST(MotionPredictionTest, PropagateFromRestIntegratesAcceleration) {
    // 1 m/s^2 along x on top of gravity, for one second
    auto samples = Samples(1.0, Eigen::Vector3d::Zero());
    for (auto& sample : samples) {
        sample.vals[3] = 1.0F;
    }
    const auto state = map::utils::PropagateFromRest(
        {0.0, 0.0, -kGravity}, samples, kKeyframeTs,
        kKeyframeTs + 1'000'000'000, gtsam::imuBias::ConstantBias{}, Camera(),
        Imu());
    EXPECT_NEAR(state.velocity_cam.x(), 1.0, 1e-4);
    EXPECT_NEAR(state.velocity_cam.y(), 0.0, 1e-6);
    EXPECT_NEAR(state.velocity_cam.z(), 0.0, 1e-4);

    // At rest, it stays there
    const auto rest = map::utils::PropagateFromRest(
        {0.0, 0.0, -kGravity}, Samples(1.0, Eigen::Vector3d::Zero()),
        kKeyframeTs, kKeyframeTs + 1'000'000'000,
        gtsam::imuBias::ConstantBias{}, Camera(), Imu());
    EXPECT_LT(rest.velocity_cam.norm(), 1e-4);
}

TEST(MotionPredictionTest, StartupStateSplitsTheBiasAlongGravity) {
    auto samples = Samples(1.0, Eigen::Vector3d{0.08, 0.0, 0.0});
    for (auto& sample : samples) {
        sample.vals[5] = 9.79F;
    }
    const auto state = map::utils::RecoverStartupImuState(
        map::utils::StationaryWindow{.samples = samples, .score = 0.0},
        Camera(), Imu(), kGravity);

    EXPECT_NEAR(state.gravity_cam.z(), -kGravity, 1e-9);
    EXPECT_NEAR(state.gravity_cam.head<2>().norm(), 0.0, 1e-9);
    EXPECT_NEAR(state.bias.accelerometer().z(), 9.79 - kGravity, 1e-4);
    EXPECT_NEAR(state.bias.gyroscope().x(), 0.08, 1e-6);
}

TEST(MotionPredictionTest, LastStationaryWindowEndsAtMotionOnset) {
    // Parked for 3 s, then shaking for 1 s
    auto samples = Samples(4.0, Eigen::Vector3d::Zero());
    const size_t onset = samples.size() * 3 / 4;
    for (size_t i = onset; i < samples.size(); ++i) {
        samples[i].vals[5] += (i % 2 == 0) ? 1.0F : -1.0F;
    }

    const auto window = map::utils::FindLastStationaryWindow(samples, 200, 0.5);
    ASSERT_TRUE(window.has_value());
    const size_t end
        = static_cast<size_t>(&window->samples.back() - samples.data() + 1);
    EXPECT_LE(end, onset);
    EXPECT_GT(end + 100, onset);

    // Fewer samples than a window
    EXPECT_FALSE(
        map::utils::FindLastStationaryWindow({samples.data(), 150}, 200, 0.5)
            .has_value());
}

}  // namespace
