#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <vector>

// Two-view epipolar geometry used by the v2 pose estimate passes. Pure
// functions over point correspondences: `prev[i]` and `curr[i]` are the same
// scene point seen in the previous and the current view. The camera is
// calibrated, so everything works on rays (undistorted normalised image plane
// coordinates) and the epipolar matrix is the essential matrix
namespace wslam::v2::util {

// Points per minimal sample of the eight-point algorithm
constexpr size_t kEightPointSampleSize = 8;

// Hartley-Zisserman normalising similarity: moves the centroid to the origin
// and scales the mean distance from it to sqrt(2). Returns T such that
// x_normalised = T * x (homogeneous). Identity scale for coincident points
[[nodiscard]] Eigen::Matrix3d NormalisingTransform(
    std::span<const Eigen::Vector2d> points);

// First-order (Sampson) approximation of the squared geometric distance of
// a correspondence to the epipolar geometry `m` (E on rays, F on pixels), in
// the squared units of the coordinates. Infinite when undefined
[[nodiscard]] double SampsonDistance(const Eigen::Matrix3d& m,
                                     const Eigen::Vector2d& prev,
                                     const Eigen::Vector2d& curr);

// Eight-point fit of the essential matrix on calibrated (normalised image
// plane) coordinates, ray_curr^T * E * ray_prev = 0. The essential
// constraint (two equal singular values, one zero) is enforced on the final,
// denormalised matrix. `weights`, when not empty, scales each
// correspondence's constraint, so a weight of 2 counts like the
// correspondence twice over. nullopt when the result is not finite
[[nodiscard]] std::optional<Eigen::Matrix3d> FitEssentialMatrix(
    std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr,
    std::span<const double> weights = {});

struct RansacOptions {
    // Probability that at least one drawn sample is all inliers when the
    // search stops. It sets the iteration count: with the best inlier ratio
    // w found so far, N = log(1 - confidence) / log(1 - w^8) samples are
    // needed, re-derived every time the best model improves. Few iterations
    // when most matches are inliers, many when few are
    double confidence = 0.99;

    // Sampson distance (squared pixels) below which a correspondence is an
    // inlier. 3.84 is the 95% chi-squared quantile for 1 DoF at 1 px noise
    double inlier_threshold = 3.84;

    // Fewer inliers than this is no model
    size_t min_inliers = kEightPointSampleSize;

    // Runtime guard only, for inlier ratios so low that N explodes (w = 0.3
    // needs ~70k samples). Stopping here means `confidence` was not reached,
    // which the result reports
    uint32_t max_iterations = 10000;

    // Least-squares re-fits on the inliers after sampling, each followed by
    // re-classifying every correspondence (see EstimateEssentialRansac)
    uint32_t refit_rounds = 3;
};

struct RansacOutcome {
    // Valid when model_found. Fitted on all of `inliers`
    Eigen::Matrix3d essential_matrix = Eigen::Matrix3d::Zero();
    // Indices of the inlier correspondences, ascending
    std::vector<size_t> inliers;
    // Samples drawn
    uint32_t iterations = 0;
    // Samples the confidence needs at the final best inlier ratio, before
    // the max_iterations guard. Zero when no model was found
    uint64_t required_iterations = 0;
    uint32_t degenerate_samples = 0;
    bool model_found = false;
    // Whether iterations reached required_iterations
    bool confidence_reached = false;
};

// Samples needed to draw at least one all-inlier sample of `sample_size`
// with probability `confidence` when `inlier_ratio` of the data are inliers
[[nodiscard]] uint64_t RequiredRansacIterations(double inlier_ratio,
                                                double confidence,
                                                size_t sample_size
                                                = kEightPointSampleSize);

// Adaptive RANSAC over eight-point essential-matrix fits on rays. The inlier
// test is the Sampson distance scaled to squared pixels by `focal_px`, so
// options.inlier_threshold is in pixels as for an image-space fit.
//
// The best sample's inliers are then re-fitted by (weighted) least squares
// and re-classified, up to options.refit_rounds times, for as long as the
// inlier set changes without shrinking. The returned matrix is always the
// least-squares fit on the returned inliers. `weights` (empty, or one per
// correspondence) only enter these re-fits: a minimal sample is an exact
// fit, which weights do not change. Fewer correspondences than
// options.min_inliers give no model
[[nodiscard]] RansacOutcome EstimateEssentialRansac(
    std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr, std::span<const double> weights,
    double focal_px, const RansacOptions& options, std::mt19937_64& rng);

// Motion taking a point from the previous camera frame to the current one:
// p_curr = r * p_prev + t
struct RelativePose {
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// Essential matrix [t]x * r of a pose
[[nodiscard]] Eigen::Matrix3d EssentialFromPose(const RelativePose& pose);

// The four (R, t) an essential matrix decomposes into (Hartley-Zisserman
// 9.6.2), t of unit length
[[nodiscard]] std::array<RelativePose, 4> DecomposeEssential(
    const Eigen::Matrix3d& e);

// Linear (DLT) triangulation of one correspondence of calibrated rays.
// Returns the point in the previous camera frame, nullopt at infinity
[[nodiscard]] std::optional<Eigen::Vector3d> TriangulatePoint(
    const RelativePose& pose, const Eigen::Vector2d& ray_prev,
    const Eigen::Vector2d& ray_curr);

struct PoseRecovery {
    RelativePose pose;
    // Per correspondence, in the previous camera frame. nullopt where the
    // triangulation failed
    std::vector<std::optional<Eigen::Vector3d>> points_prev;
    // Points in front of both cameras under `pose`
    size_t in_front = 0;
};

// Triangulates every correspondence under `pose` and counts the points in
// front of both cameras
[[nodiscard]] PoseRecovery TriangulateAll(
    const RelativePose& pose, std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr);

// Picks the decomposition of `e` with the most points in front of both
// cameras. nullopt when no candidate puts any point in front of both
[[nodiscard]] std::optional<PoseRecovery> RecoverRelativePose(
    const Eigen::Matrix3d& e, std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr);

struct RefineOptions {
    // Residuals beyond this many pixels are down-weighted (Huber)
    double huber_px = 1.0;
    uint32_t max_iterations = 20;
    // Stops when a step lowers the cost by less than this fraction
    double relative_tolerance = 1e-10;
};

struct RefineOutcome {
    RelativePose pose;
    // Robust weighted cost (squared pixels) before and after
    double initial_cost = 0.0;
    double final_cost = 0.0;
    // Steps taken
    uint32_t iterations = 0;
};

// Levenberg-Marquardt refinement of (R, t), t on the unit sphere (5 DoF),
// minimising the robust weighted sum of squared Sampson errors of
// E = [t]x * R over the correspondences, in pixels via `focal_px`. `weights`
// is empty or one per correspondence. Only cost-lowering steps are taken,
// so final_cost <= initial_cost
[[nodiscard]] RefineOutcome RefineRelativePose(
    const RelativePose& pose, std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr, std::span<const double> weights,
    double focal_px, const RefineOptions& options = {});

}  // namespace wslam::v2::util
