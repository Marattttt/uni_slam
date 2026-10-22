#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <vector>

#include "data/provider_base.hpp"
#include "map/map_helpers_lods.hpp"
#include "pose_estimate_v2/util/camera.hpp"
#include "pose_estimate_v2/util/epipolar_geometry.hpp"

using namespace wslam;
namespace util = wslam::v2::util;

namespace {

constexpr double kDegToRad = std::numbers::pi / 180.0;
constexpr double kImageWidth = 752.0;
constexpr double kImageHeight = 480.0;
// Mean focal length of EurocCamera()
constexpr double kFocalPx = 0.5 * (458.654 + 457.296);

// EuRoC cam0
data::CamSensorParams EurocCamera() {
    data::CamSensorParams camera{};
    camera.resolution = {752, 480};
    camera.intrinsics = {458.654, 457.296, 367.215, 248.375};
    camera.distortion_coefficients
        = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
    return camera;
}

double RotationErrorDeg(const Eigen::Matrix3d& estimate,
                        const Eigen::Matrix3d& truth) {
    return map::util::ComputeRotationAngle(estimate * truth.transpose())
           / kDegToRad;
}

double DirectionErrorDeg(const Eigen::Vector3d& estimate,
                         const Eigen::Vector3d& truth) {
    const double cosine
        = std::clamp(estimate.normalized().dot(truth.normalized()), -1.0, 1.0);
    return std::acos(cosine) / kDegToRad;
}

// Fixed seeds keep the tests deterministic
std::mt19937_64 FixedRng(unsigned seed = 1) {
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    return std::mt19937_64(seed);
}

struct SceneOptions {
    size_t points = 300;
    // Scales the 0.3 m baseline. At 1 the median disparity from translation
    // is ~27 px, a low-parallax pair
    double baseline_scale = 1.0;
    double noise_px = 0.0;
    double outlier_ratio = 0.0;
    unsigned seed = 7;
};

// Points 2-8 m in front of the previous camera, seen by both cameras of a
// known relative pose. Pixels are undistorted (ideal pinhole), rays are
// K^-1 * pixel
struct Scene {
    util::RelativePose truth;
    std::vector<Eigen::Vector3d> points_prev;
    std::vector<Eigen::Vector2d> prev_px;
    std::vector<Eigen::Vector2d> curr_px;
    std::vector<Eigen::Vector2d> prev_ray;
    std::vector<Eigen::Vector2d> curr_ray;
    std::vector<bool> is_inlier;
};

Eigen::Vector2d ToRay(const Eigen::Vector2d& px,
                      const data::CamSensorParams& camera) {
    const Eigen::Vector4d& k = camera.intrinsics;
    return {(px.x() - k[2]) / k[0], (px.y() - k[3]) / k[1]};
}

bool InImage(const Eigen::Vector2d& px) {
    return px.x() >= 0.0 && px.y() >= 0.0 && px.x() < kImageWidth
           && px.y() < kImageHeight;
}

Scene MakeScene(const SceneOptions& options) {
    const auto camera = EurocCamera();

    Scene scene;
    scene.truth.r
        = Eigen::AngleAxisd(5.0 * kDegToRad,
                            Eigen::Vector3d(0.2, 1.0, 0.1).normalized())
              .toRotationMatrix();
    scene.truth.t = options.baseline_scale * Eigen::Vector3d(0.25, 0.05, 0.15);

    std::mt19937_64 rng = FixedRng(options.seed);
    std::uniform_real_distribution<double> lateral(-4.0, 4.0);
    std::uniform_real_distribution<double> depth(2.0, 8.0);
    // A zero deviation is invalid; noise is only drawn when noise_px > 0
    std::normal_distribution<double> noise(
        0.0, options.noise_px > 0.0 ? options.noise_px : 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    const auto project = [&](const Eigen::Vector3d& point) {
        return util::NormalisedToPixel(point.head<2>() / point.z(), camera);
    };

    while (scene.points_prev.size() < options.points) {
        const Eigen::Vector3d point(lateral(rng), lateral(rng), depth(rng));
        const Eigen::Vector3d point_curr
            = scene.truth.r * point + scene.truth.t;
        if (point_curr.z() <= 0.0) {
            continue;
        }

        Eigen::Vector2d prev = project(point);
        Eigen::Vector2d curr = project(point_curr);
        if (!InImage(prev) || !InImage(curr)) {
            continue;
        }

        if (options.noise_px > 0.0) {
            prev += Eigen::Vector2d(noise(rng), noise(rng));
            curr += Eigen::Vector2d(noise(rng), noise(rng));
        }

        bool inlier = true;
        if (unit(rng) < options.outlier_ratio) {
            // A random pixel well away from the true one
            Eigen::Vector2d random;
            do {
                random = {unit(rng) * kImageWidth, unit(rng) * kImageHeight};
            } while ((random - curr).norm() < 30.0);
            curr = random;
            inlier = false;
        }

        scene.points_prev.push_back(point);
        scene.prev_px.push_back(prev);
        scene.curr_px.push_back(curr);
        scene.prev_ray.push_back(ToRay(prev, camera));
        scene.curr_ray.push_back(ToRay(curr, camera));
        scene.is_inlier.push_back(inlier);
    }

    return scene;
}

}  // namespace

TEST(EpipolarGeometryTest, EssentialMatrixFitsNoiseFreeCorrespondences) {
    const Scene scene = MakeScene({.points = 50});

    for (const size_t count : {size_t{8}, scene.prev_ray.size()}) {
        const auto prev = std::span{scene.prev_ray}.first(count);
        const auto curr = std::span{scene.curr_ray}.first(count);

        const auto e = util::FitEssentialMatrix(prev, curr);
        ASSERT_TRUE(e.has_value()) << count << " points";

        const Eigen::Vector3d singular_values
            = Eigen::JacobiSVD<Eigen::Matrix3d>(*e).singularValues();
        EXPECT_NEAR(singular_values(0), singular_values(1), 1e-9)
            << "E must have two equal singular values";
        EXPECT_LT(singular_values(2), 1e-9 * singular_values(0))
            << "E must have rank 2";

        // Every correspondence of the scene, not only the fitted ones
        for (size_t i = 0; i < scene.prev_ray.size(); ++i) {
            EXPECT_LT(
                util::SampsonDistance(*e, scene.prev_ray[i], scene.curr_ray[i]),
                1e-14)
                << "point " << i << " fitted on " << count;
        }
    }
}

TEST(EpipolarGeometryTest, UniformWeightsLeaveTheFitUnchanged) {
    const Scene scene = MakeScene({.points = 100, .noise_px = 0.5});

    const auto unweighted
        = util::FitEssentialMatrix(scene.prev_ray, scene.curr_ray);
    const std::vector<double> weights(scene.prev_ray.size(), 0.7);
    const auto weighted
        = util::FitEssentialMatrix(scene.prev_ray, scene.curr_ray, weights);
    ASSERT_TRUE(unweighted.has_value() && weighted.has_value());

    // E is defined up to sign
    const double sign
        = unweighted->cwiseProduct(*weighted).sum() < 0.0 ? -1.0 : 1.0;
    EXPECT_LT((*unweighted - sign * *weighted).norm(), 1e-9);
}

TEST(EpipolarGeometryTest, DownWeightingNoisyCorrespondencesHelps) {
    // A third of the correspondences are much noisier than the rest; their
    // low weight should keep them from dominating the fit
    const Scene clean = MakeScene({.points = 300, .noise_px = 0.3});
    const Scene noisy = MakeScene({.points = 300, .noise_px = 4.0});

    std::vector<Eigen::Vector2d> prev;
    std::vector<Eigen::Vector2d> curr;
    std::vector<double> weights;
    for (size_t i = 0; i < clean.prev_ray.size(); ++i) {
        const bool is_noisy = i % 3 == 0;
        const Scene& source = is_noisy ? noisy : clean;
        prev.push_back(source.prev_ray[i]);
        curr.push_back(source.curr_ray[i]);
        weights.push_back(is_noisy ? 0.1 : 1.0);
    }

    const auto pose_error = [&](std::span<const double> fit_weights) {
        const auto e = util::FitEssentialMatrix(prev, curr, fit_weights);
        EXPECT_TRUE(e.has_value());
        const auto recovery = util::RecoverRelativePose(*e, prev, curr);
        EXPECT_TRUE(recovery.has_value());
        return DirectionErrorDeg(recovery->pose.t, clean.truth.t);
    };

    EXPECT_LT(pose_error(weights), 0.5 * pose_error({}));
}

TEST(EpipolarGeometryTest, RequiredIterationsFollowTheInlierRatio) {
    EXPECT_EQ(util::RequiredRansacIterations(1.0, 0.99), 1U);
    // log(0.01) / log(1 - 0.5^8) = 1176.6
    EXPECT_EQ(util::RequiredRansacIterations(0.5, 0.99), 1177U);
    // log(0.01) / log(1 - 0.75^8) = 43.7
    EXPECT_EQ(util::RequiredRansacIterations(0.75, 0.99), 44U);
    EXPECT_EQ(util::RequiredRansacIterations(0.0, 0.99),
              std::numeric_limits<uint64_t>::max());

    // 0.01^8 = 1e-16 rounds 1 - p to 1; the bound must stay finite
    const uint64_t tiny = util::RequiredRansacIterations(0.01, 0.99);
    EXPECT_GT(tiny, uint64_t{4'000'000'000'000'000});
    EXPECT_LT(tiny, std::numeric_limits<uint64_t>::max());

    // Monotonic: fewer inliers need more samples
    uint64_t previous = 1;
    for (const double ratio : {0.95, 0.8, 0.6, 0.4, 0.3}) {
        const uint64_t required = util::RequiredRansacIterations(ratio, 0.99);
        EXPECT_GT(required, previous) << "ratio " << ratio;
        previous = required;
    }
}

TEST(EpipolarGeometryTest, RansacSeparatesInliersFromOutliers) {
    // A 1.2 m baseline. At low parallax the linear eight-point re-fit is
    // weakly constrained, and a single gross outlier that the best sample
    // model happens to accept can bend it enough to lose ~6% of the true
    // inliers (seen at baseline_scale 1; benchmarks/POSE_ESTIMATE_V2_NOTE.md)
    const Scene scene = MakeScene({.points = 300,
                                   .baseline_scale = 4.0,
                                   .noise_px = 0.5,
                                   .outlier_ratio = 0.25});

    std::mt19937_64 rng = FixedRng();
    const util::RansacOutcome outcome
        = util::EstimateEssentialRansac(scene.prev_ray, scene.curr_ray, {},
                                        kFocalPx, util::RansacOptions{}, rng);
    ASSERT_TRUE(outcome.model_found);
    EXPECT_TRUE(outcome.confidence_reached);

    const auto true_inliers
        = static_cast<size_t>(std::ranges::count(scene.is_inlier, true));
    size_t found_true_inliers = 0;
    for (const size_t index : outcome.inliers) {
        found_true_inliers += scene.is_inlier.at(index) ? 1U : 0U;
    }

    const double recall = static_cast<double>(found_true_inliers)
                          / static_cast<double>(true_inliers);
    const double precision = static_cast<double>(found_true_inliers)
                             / static_cast<double>(outcome.inliers.size());
    EXPECT_GE(recall, 0.95);
    EXPECT_GE(precision, 0.95);
}

TEST(EpipolarGeometryTest, RansacIterationCountAdaptsToTheInlierRatio) {
    const auto iterations_for = [](double outlier_ratio, double noise_px) {
        const Scene scene = MakeScene({.points = 300,
                                       .noise_px = noise_px,
                                       .outlier_ratio = outlier_ratio});
        std::mt19937_64 rng = FixedRng();
        return util::EstimateEssentialRansac(scene.prev_ray, scene.curr_ray, {},
                                             kFocalPx, util::RansacOptions{},
                                             rng);
    };

    // Noise-free and no outliers: the first sample explains every point
    const auto exact = iterations_for(0.0, 0.0);
    ASSERT_TRUE(exact.model_found);
    EXPECT_TRUE(exact.confidence_reached);
    EXPECT_EQ(exact.iterations, 1U);

    // With noise a minimal sample's model misses some inliers, so the
    // observed ratio is below 1 and a few more samples are drawn
    const auto clean = iterations_for(0.0, 0.5);
    ASSERT_TRUE(clean.model_found);
    EXPECT_TRUE(clean.confidence_reached);
    EXPECT_GE(clean.iterations, exact.iterations);

    const auto some_outliers = iterations_for(0.25, 0.5);
    ASSERT_TRUE(some_outliers.model_found);
    EXPECT_TRUE(some_outliers.confidence_reached);
    EXPECT_GT(some_outliers.iterations, clean.iterations);

    // Half outliers need over a thousand samples, past the old fixed 500
    const auto many_outliers = iterations_for(0.5, 0.5);
    ASSERT_TRUE(many_outliers.model_found);
    EXPECT_TRUE(many_outliers.confidence_reached);
    EXPECT_GT(many_outliers.iterations, 500U);
    EXPECT_GE(many_outliers.iterations, many_outliers.required_iterations);
}

TEST(EpipolarGeometryTest, RansacReportsTheIterationGuard) {
    const Scene scene
        = MakeScene({.points = 300, .noise_px = 0.5, .outlier_ratio = 0.6});

    std::mt19937_64 rng = FixedRng();
    const util::RansacOutcome outcome = util::EstimateEssentialRansac(
        scene.prev_ray, scene.curr_ray, {}, kFocalPx,
        util::RansacOptions{.max_iterations = 50}, rng);

    EXPECT_EQ(outcome.iterations, 50U);
    if (outcome.model_found) {
        EXPECT_FALSE(outcome.confidence_reached);
        EXPECT_GT(outcome.required_iterations, 50U);
    }
}

TEST(EpipolarGeometryTest, RansacNeedsEnoughCorrespondences) {
    const Scene scene = MakeScene({.points = 7});

    std::mt19937_64 rng = FixedRng();
    const util::RansacOutcome outcome
        = util::EstimateEssentialRansac(scene.prev_ray, scene.curr_ray, {},
                                        kFocalPx, util::RansacOptions{}, rng);

    EXPECT_FALSE(outcome.model_found);
    EXPECT_EQ(outcome.iterations, 0U);
    EXPECT_TRUE(outcome.inliers.empty());
}

TEST(EpipolarGeometryTest, OneEssentialDecompositionIsTheTruePose) {
    const Scene scene = MakeScene({.points = 50});

    const auto e = util::FitEssentialMatrix(scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(e.has_value());

    size_t matching = 0;
    for (const util::RelativePose& candidate : util::DecomposeEssential(*e)) {
        EXPECT_NEAR(candidate.r.determinant(), 1.0, 1e-9);
        EXPECT_NEAR(candidate.t.norm(), 1.0, 1e-9);
        if (RotationErrorDeg(candidate.r, scene.truth.r) < 1e-6
            && DirectionErrorDeg(candidate.t, scene.truth.t) < 1e-6) {
            ++matching;
        }
    }
    EXPECT_EQ(matching, 1U);
}

TEST(EpipolarGeometryTest, TriangulatesNoiseFreePointsExactly) {
    const Scene scene = MakeScene({.points = 50});

    for (size_t i = 0; i < scene.points_prev.size(); ++i) {
        const auto point = util::TriangulatePoint(
            scene.truth, scene.prev_ray[i], scene.curr_ray[i]);
        ASSERT_TRUE(point.has_value()) << "point " << i;
        EXPECT_LT((*point - scene.points_prev[i]).norm(),
                  1e-9 * scene.points_prev[i].norm())
            << "point " << i;
    }
}

TEST(EpipolarGeometryTest, RecoversTheNoiseFreePose) {
    const Scene scene = MakeScene({.points = 100});

    const auto e = util::FitEssentialMatrix(scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(e.has_value());
    const auto recovery
        = util::RecoverRelativePose(*e, scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(recovery.has_value());

    EXPECT_LT(RotationErrorDeg(recovery->pose.r, scene.truth.r), 1e-6);
    EXPECT_LT(DirectionErrorDeg(recovery->pose.t, scene.truth.t), 1e-6);
    EXPECT_EQ(recovery->in_front, scene.points_prev.size());

    // Triangulated with unit translation: the truth scaled by 1 / |t|
    const double scale = 1.0 / scene.truth.t.norm();
    for (size_t i = 0; i < scene.points_prev.size(); ++i) {
        ASSERT_TRUE(recovery->points_prev[i].has_value());
        EXPECT_LT(
            (*recovery->points_prev[i] - scale * scene.points_prev[i]).norm(),
            1e-6 * scale * scene.points_prev[i].norm());
    }
}

TEST(EpipolarGeometryTest, RecoversTheNoisyPose) {
    const Scene scene = MakeScene({.points = 300, .noise_px = 0.5});

    const auto e = util::FitEssentialMatrix(scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(e.has_value());
    const auto recovery
        = util::RecoverRelativePose(*e, scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(recovery.has_value());

    EXPECT_LT(RotationErrorDeg(recovery->pose.r, scene.truth.r), 0.2);
    EXPECT_LT(DirectionErrorDeg(recovery->pose.t, scene.truth.t), 2.0);
}

TEST(EpipolarGeometryTest, RefinementKeepsAnExactPose) {
    const Scene scene = MakeScene({.points = 100});
    const util::RelativePose truth{.r = scene.truth.r,
                                   .t = scene.truth.t.normalized()};

    const util::RefineOutcome outcome = util::RefineRelativePose(
        truth, scene.prev_ray, scene.curr_ray, {}, kFocalPx);

    EXPECT_LT(outcome.initial_cost, 1e-12);
    EXPECT_LE(outcome.final_cost, outcome.initial_cost);
    EXPECT_LT(RotationErrorDeg(outcome.pose.r, truth.r), 1e-6);
    EXPECT_LT(DirectionErrorDeg(outcome.pose.t, truth.t), 1e-6);
}

TEST(EpipolarGeometryTest, RefinementConvergesFromAPerturbedPose) {
    const Scene scene = MakeScene({.points = 300});
    const util::RelativePose start{
        .r = Eigen::AngleAxisd(1.0 * kDegToRad, Eigen::Vector3d::UnitY())
                 .toRotationMatrix()
             * scene.truth.r,
        .t = (scene.truth.t.normalized() + Eigen::Vector3d(0.0, 0.08, 0.0))
                 .normalized(),
    };

    const util::RefineOutcome outcome = util::RefineRelativePose(
        start, scene.prev_ray, scene.curr_ray, {}, kFocalPx);

    EXPECT_LT(outcome.final_cost, 1e-6 * outcome.initial_cost);
    EXPECT_LT(RotationErrorDeg(outcome.pose.r, scene.truth.r), 1e-4);
    EXPECT_LT(DirectionErrorDeg(outcome.pose.t, scene.truth.t), 1e-3);
}

TEST(EpipolarGeometryTest, RefinementLowersTheNoisyCost) {
    const Scene scene = MakeScene({.points = 300, .noise_px = 1.0});

    const auto e = util::FitEssentialMatrix(scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(e.has_value());
    const auto recovery
        = util::RecoverRelativePose(*e, scene.prev_ray, scene.curr_ray);
    ASSERT_TRUE(recovery.has_value());

    const util::RefineOutcome outcome = util::RefineRelativePose(
        recovery->pose, scene.prev_ray, scene.curr_ray, {}, kFocalPx);

    EXPECT_GT(outcome.iterations, 0U);
    EXPECT_LT(outcome.final_cost, outcome.initial_cost);
    EXPECT_NEAR(outcome.pose.r.determinant(), 1.0, 1e-9);
    EXPECT_NEAR(outcome.pose.t.norm(), 1.0, 1e-9);
    // Still the right pose, not a drift away from it
    EXPECT_LT(RotationErrorDeg(outcome.pose.r, scene.truth.r), 0.2);
    EXPECT_LT(DirectionErrorDeg(outcome.pose.t, scene.truth.t), 2.0);
}

TEST(CameraModelTest, UndistortionInvertsProjection) {
    const auto camera = EurocCamera();

    // Across the image, corners included, where the distortion is largest
    for (const double x : {-0.75, -0.4, 0.0, 0.3, 0.75}) {
        for (const double y : {-0.5, -0.2, 0.0, 0.25, 0.5}) {
            const Eigen::Vector3d point(x, y, 1.0);
            const Eigen::Vector2d px = util::ProjectToPixel(point, camera);
            ASSERT_TRUE(px.allFinite());

            const Eigen::Vector2d ray = map::util::PixelToNormalised(
                px, camera.intrinsics, camera.distortion_coefficients, 20,
                1e-12);
            EXPECT_LT((ray - point.head<2>()).norm(), 1e-6)
                << "at (" << x << ", " << y << ")";
        }
    }

    EXPECT_FALSE(util::ProjectToPixel(Eigen::Vector3d(0.0, 0.0, -1.0), camera)
                     .allFinite());
}

namespace {
// UZH-FPV indoor_45 Snapdragon cam0 (Kalibr), a ~170 degree fisheye
data::CamSensorParams UzhFpvCamera() {
    data::CamSensorParams camera{};
    camera.resolution = {640, 480};
    camera.intrinsics = {275.46015578667294, 274.9948095922592,
                         315.958384100568, 242.7123497822731};
    camera.distortion_model = data::DistortionModel::Equidistant;
    camera.distortion_coefficients
        = {-6.545154718304953e-06, -0.010379525898159981, 0.014935312423953146,
           -0.005639061406567785};
    return camera;
}
}  // namespace

TEST(CameraModelTest, EquidistantUndistortionInvertsProjection) {
    const auto camera = UzhFpvCamera();

    // Out to ~80 degrees off the axis (|xy| = 5.7), the image corners
    for (const double x : {-4.0, -1.5, -0.3, 0.0, 0.6, 2.0, 4.0}) {
        for (const double y : {-4.0, -1.0, 0.0, 0.4, 4.0}) {
            const Eigen::Vector3d point(x, y, 1.0);
            const Eigen::Vector2d px = util::ProjectToPixel(point, camera);
            ASSERT_TRUE(px.allFinite());

            const Eigen::Vector2d ray
                = map::util::PixelToNormalised(px, camera, 20, 1e-12);
            EXPECT_LT((ray - point.head<2>()).norm(), 1e-6 * point.norm())
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST(CameraModelTest, EquidistantMatchesTheKalibrModel) {
    const auto camera = UzhFpvCamera();
    const Eigen::Vector4d& d = camera.distortion_coefficients;

    // theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    const Eigen::Vector2d xy(0.8, -0.6);
    const double theta = std::atan(xy.norm());
    const double theta_d
        = theta
          * (1 + d[0] * std::pow(theta, 2) + d[1] * std::pow(theta, 4)
             + d[2] * std::pow(theta, 6) + d[3] * std::pow(theta, 8));
    const Eigen::Vector2d expected = xy.normalized() * theta_d;

    const Eigen::Vector2d distorted = map::util::DistortNormalised(xy, camera);
    EXPECT_LT((distorted - expected).norm(), 1e-12);
    // The radial-tangential overload must not be picked for this camera
    EXPECT_GT((map::util::DistortNormalised(xy, camera.distortion_coefficients)
               - expected)
                  .norm(),
              1e-3);

    // The principal point maps to the optical axis
    const Eigen::Vector2d centre = map::util::PixelToNormalised(
        camera.intrinsics.tail<2>(), camera, 20, 1e-12);
    EXPECT_LT(centre.norm(), 1e-12);
}

TEST(CameraModelTest, ParsesDistortionModelNames) {
    EXPECT_EQ(data::ParseDistortionModel("radial-tangential"),
              data::DistortionModel::RadialTangential);
    EXPECT_EQ(data::ParseDistortionModel("radtan"),
              data::DistortionModel::RadialTangential);
    EXPECT_EQ(data::ParseDistortionModel("equidistant"),
              data::DistortionModel::Equidistant);
    EXPECT_FALSE(data::ParseDistortionModel("omni").has_value());
    EXPECT_EQ(data::ToString(data::DistortionModel::Equidistant),
              "equidistant");
}
