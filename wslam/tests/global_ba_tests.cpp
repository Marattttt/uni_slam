// The export-time global bundle adjustment on a synthetic visual-inertial
// run: a camera circling inside a ring of landmarks, a consistent IMU
// stream, and an initial guess that drifted in scale and yaw

#include <gtest/gtest.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "data/provider_base.hpp"
#include "map/utils.hpp"
#include "map_v2/loop_closure/global_ba.hpp"

namespace lc = wslam::v2::mapping::loop_closure;

namespace {

constexpr double kRadius = 2.0;       // m, camera circle
constexpr double kAngularRate = 0.5;  // rad/s
// Over 2 pi: the end revisits the start's landmarks, as a loop closure
// would, which observes the yaw drift a constant gyroscope bias makes
constexpr double kDuration = 14.0;  // s
constexpr double kKeyframePeriod = 0.25;
constexpr uint64_t kImuPeriodNs = 5'000'000;
constexpr double kGravity = 9.81;

uint64_t ToNs(double seconds) {
    return static_cast<uint64_t>(std::llround(seconds * 1e9));
}

// Camera on the circle, looking outwards, y down
gtsam::Pose3 TruePose(double t) {
    const double angle = kAngularRate * t;
    const Eigen::Vector3d outward{std::cos(angle), std::sin(angle), 0.0};
    const Eigen::Vector3d down{0.0, 0.0, -1.0};
    Eigen::Matrix3d rotation;
    rotation.col(2) = outward;
    rotation.col(1) = down;
    rotation.col(0) = down.cross(outward);
    const Eigen::Vector3d position
        = kRadius * outward + Eigen::Vector3d{0.0, 0.0, 0.2 * std::sin(t)};
    return {gtsam::Rot3(rotation), gtsam::Point3(position)};
}

Eigen::Vector3d TrueVelocity(double t) {
    const double angle = kAngularRate * t;
    return {-kRadius * kAngularRate * std::sin(angle),
            kRadius * kAngularRate * std::cos(angle), 0.2 * std::cos(t)};
}

Eigen::Vector3d TrueAcceleration(double t) {
    const double angle = kAngularRate * t;
    const double w2 = kAngularRate * kAngularRate;
    return {-kRadius * w2 * std::cos(angle), -kRadius * w2 * std::sin(angle),
            -0.2 * std::sin(t)};
}

// Body = camera = IMU. Each sample covers the period before its timestamp,
// so it carries the period's midpoint acceleration, in the orientation the
// preintegration applies it in: the period's start
std::vector<wslam::data::IMUReading> ImuStream() {
    const Eigen::Vector3d gravity{0.0, 0.0, -kGravity};
    std::vector<wslam::data::IMUReading> samples;
    for (uint64_t ns = kImuPeriodNs; ns <= ToNs(kDuration);
         ns += kImuPeriodNs) {
        const double start = static_cast<double>(ns - kImuPeriodNs) * 1e-9;
        const double t = start + 0.5e-9 * static_cast<double>(kImuPeriodNs);
        const Eigen::Matrix3d r = TruePose(start).rotation().matrix();
        const Eigen::Vector3d rate
            = r.transpose() * Eigen::Vector3d{0.0, 0.0, kAngularRate};
        const Eigen::Vector3d force
            = r.transpose() * (TrueAcceleration(t) - gravity);
        wslam::data::IMUReading reading{.timestamp = ns, .vals = {}};
        reading.vals << static_cast<float>(rate.x()),
            static_cast<float>(rate.y()), static_cast<float>(rate.z()),
            static_cast<float>(force.x()), static_cast<float>(force.y()),
            static_cast<float>(force.z());
        samples.push_back(reading);
    }
    return samples;
}

wslam::data::CamSensorParams Camera() {
    return {
        .T_BS = Eigen::Matrix4d::Identity(),
        .rate_hz = 20,
        .resolution = {752, 480},
        .camera_model = "pinhole",
        .intrinsics = {450.0, 450.0, 376.0, 240.0},
        .distortion_model = "radial-tangential",
        .distortion_coefficients = Eigen::Vector4d::Zero(),
    };
}

wslam::data::IMUSensorParams Imu() {
    return {
        .T_BS = Eigen::Matrix4d::Identity(),
        .rate_hz = 200,
        .gyroscope_noise_density = 1.6968e-04,
        .gyroscope_random_walk = 1.9393e-05,
        .accelerometer_noise_density = 2.0e-3,
        .accelerometer_random_walk = 3.0e-3,
    };
}

struct Scenario {
    lc::GlobalBaInput input;
    std::vector<gtsam::Pose3> truth;
};

// The initial guess drifts: positions grow by drift_scale over the run and
// the heading turns by drift_yaw_rad, as an uncorrected VIO would
Scenario MakeScenario(double drift_scale, double drift_yaw_rad) {
    Scenario scenario;
    auto& input = scenario.input;
    input.calibration
        = std::make_shared<gtsam::Cal3_S2>(450.0, 450.0, 0.0, 376.0, 240.0);
    input.imu_params = wslam::map::utils::CreateImuCalibration(
        Camera(), Imu(), {0.0, 0.0, -kGravity}, 1e-4);

    const auto samples = ImuStream();
    const auto keyframe_count
        = static_cast<size_t>(kDuration / kKeyframePeriod);

    std::mt19937 random(7);
    std::normal_distribution<double> pixel_noise(0.0, 0.5);
    std::normal_distribution<double> point_noise(0.0, 0.05);

    for (size_t i = 0; i < keyframe_count; ++i) {
        const double t = static_cast<double>(i) * kKeyframePeriod;
        const gtsam::Pose3 truth = TruePose(t);
        scenario.truth.push_back(truth);

        const double progress
            = static_cast<double>(i) / static_cast<double>(keyframe_count - 1);
        const gtsam::Rot3 yaw = gtsam::Rot3::Rz(drift_yaw_rad * progress);
        const gtsam::Point3 origin = TruePose(0.0).translation();
        const gtsam::Point3 drifted
            = origin
              + yaw.rotate((truth.translation() - origin)
                           * (1.0 + (drift_scale - 1.0) * progress));
        input.keyframes.push_back(lc::GlobalBaInput::Keyframe{
            .id = i + 1,
            .pose = gtsam::Pose3(yaw * truth.rotation(), drifted),
            .velocity = yaw.rotate(gtsam::Point3(TrueVelocity(t))),
            .bias = {},
        });

        if (i > 0) {
            const uint64_t from_ts = ToNs(t - kKeyframePeriod);
            const uint64_t to_ts = ToNs(t);
            lc::GlobalBaInput::ImuEdge edge{
                .from = i, .to = i + 1, .from_ts = from_ts, .to_ts = to_ts};
            for (const auto& sample : samples) {
                if (sample.timestamp > from_ts && sample.timestamp <= to_ts) {
                    edge.samples.push_back(sample);
                }
            }
            input.imu_edges.push_back(std::move(edge));
        }
    }

    // A ring of landmarks 5 to 7 m from the circle's centre
    constexpr size_t kLandmarks = 400;
    for (size_t j = 0; j < kLandmarks; ++j) {
        const double angle = 2.0 * M_PI * static_cast<double>(j) / kLandmarks;
        const double distance = 5.0 + 2.0 * static_cast<double>(j % 7) / 6.0;
        const double height = -1.0 + 2.0 * static_cast<double>(j % 11) / 10.0;
        const Eigen::Vector3d point{distance * std::cos(angle),
                                    distance * std::sin(angle), height};

        lc::GlobalBaInput::Landmark landmark{.id = j + 1, .position = {}};
        // Initial guess: seen from the first keyframe that observes it,
        // carried by that keyframe's drift
        std::optional<size_t> first_view;
        for (size_t i = 0; i < scenario.truth.size(); ++i) {
            const gtsam::PinholeCamera<gtsam::Cal3_S2> camera{
                scenario.truth[i], *input.calibration};
            const auto [pixel, in_front] = camera.projectSafe(point);
            if (!in_front || pixel.x() < 0 || pixel.x() > 752 || pixel.y() < 0
                || pixel.y() > 480) {
                continue;
            }
            first_view = first_view.value_or(i);
            landmark.observations.emplace_back(
                i + 1, pixel
                           + Eigen::Vector2d{pixel_noise(random),
                                             pixel_noise(random)});
        }
        if (!first_view) {
            continue;
        }
        const gtsam::Pose3 drift = input.keyframes[*first_view].pose
                                   * scenario.truth[*first_view].inverse();
        landmark.position
            = drift.transformFrom(gtsam::Point3(point))
              + Eigen::Vector3d{point_noise(random), point_noise(random),
                                point_noise(random)};
        input.landmarks.push_back(std::move(landmark));
    }
    return scenario;
}

double PositionRmse(
    const std::vector<gtsam::Pose3>& truth,
    const std::vector<std::pair<uint64_t, gtsam::Pose3>>& poses) {
    double sum = 0.0;
    for (const auto& [id, pose] : poses) {
        sum += (pose.translation() - truth[id - 1].translation()).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(poses.size()));
}

TEST(GlobalBaTest, RemovesScaleAndYawDrift) {
    const Scenario scenario = MakeScenario(1.03, 3.0 * M_PI / 180.0);

    std::vector<std::pair<uint64_t, gtsam::Pose3>> initial;
    for (const auto& keyframe : scenario.input.keyframes) {
        initial.emplace_back(keyframe.id, keyframe.pose);
    }
    const double initial_rmse = PositionRmse(scenario.truth, initial);

    const auto result = lc::RunGlobalBa(scenario.input, lc::GlobalBaOptions{});
    ASSERT_TRUE(result.has_value()) << result.error();

    const double final_rmse = PositionRmse(scenario.truth, result->poses);
    EXPECT_LT(final_rmse, initial_rmse / 5.0)
        << "initial " << initial_rmse << " m, final " << final_rmse << " m";
    EXPECT_LT(final_rmse, 0.02);
    EXPECT_LT(result->final_error, result->initial_error);
    EXPECT_GT(result->imu_factors, 0U);
    EXPECT_GT(result->landmarks.size(), 300U);
}

TEST(GlobalBaTest, LeavesAConsistentMapAlone) {
    const Scenario scenario = MakeScenario(1.0, 0.0);
    const auto result = lc::RunGlobalBa(scenario.input, lc::GlobalBaOptions{});
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_LT(PositionRmse(scenario.truth, result->poses), 0.01);
    EXPECT_NEAR(result->path_length_ratio, 1.0, 0.01);
}

TEST(GlobalBaTest, DropsLandmarksTheOutliersLeaveTooFewViews) {
    Scenario scenario = MakeScenario(1.0, 0.0);
    // Every fifth landmark keeps three views and loses one to a gross
    // outlier, so the second solve leaves it out
    std::vector<uint64_t> corrupted;
    for (auto& landmark : scenario.input.landmarks) {
        if (landmark.id % 5 == 0 && landmark.observations.size() >= 3) {
            landmark.observations.resize(3);
            landmark.observations.front().second += Eigen::Vector2d{60.0, 0.0};
            corrupted.push_back(landmark.id);
        }
    }
    ASSERT_FALSE(corrupted.empty());

    const auto result = lc::RunGlobalBa(scenario.input, lc::GlobalBaOptions{});
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_GE(result->outliers_removed, corrupted.size());
    // With three views a solve can spread the outlier so that another view
    // is the one dropped; most corrupted landmarks still leave
    const auto survivors = std::ranges::count_if(
        result->landmarks, [&](const auto& landmark) {
            return std::ranges::contains(corrupted, landmark.first);
        });
    EXPECT_LT(static_cast<size_t>(survivors), corrupted.size() / 2)
        << survivors << " of " << corrupted.size() << " corrupted landmarks kept";
}

TEST(GlobalBaTest, RejectsTooFewKeyframes) {
    lc::GlobalBaInput input;
    input.keyframes.resize(1);
    EXPECT_FALSE(lc::RunGlobalBa(input, lc::GlobalBaOptions{}).has_value());
}

}  // namespace
