#include "epipolar_geometry.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "assert.hpp"

namespace wslam::v2::util {

namespace {
Eigen::Vector2d Transform(const Eigen::Matrix3d& t, const Eigen::Vector2d& p) {
    return (t * p.homogeneous()).hnormalized();
}

// Least-squares solution M of curr^T * M * prev = 0 after applying the
// normalising transforms, as the right singular vector of the stacked
// constraints with the smallest singular value. Returned in normalised space
Eigen::Matrix3d SolveEpipolarConstraint(std::span<const Eigen::Vector2d> prev,
                                        std::span<const Eigen::Vector2d> curr,
                                        const Eigen::Matrix3d& t_prev,
                                        const Eigen::Matrix3d& t_curr) {
    Eigen::MatrixXd a(static_cast<Eigen::Index>(prev.size()), 9);
    for (size_t i = 0; i < prev.size(); ++i) {
        const Eigen::Vector2d p = Transform(t_prev, prev[i]);
        const Eigen::Vector2d c = Transform(t_curr, curr[i]);
        a.row(static_cast<Eigen::Index>(i)) << c.x() * p.x(), c.x() * p.y(),
            c.x(), c.y() * p.x(), c.y() * p.y(), c.y(), p.x(), p.y(), 1.0;
    }

    // Only V is needed; U would be an N x N matrix
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
    const Eigen::VectorXd m = svd.matrixV().col(8);

    Eigen::Matrix3d out;
    // Row-major reshape of the 9-vector
    // NOLINTBEGIN(readability-magic-numbers)
    // clang-format off
    out << m(0), m(1), m(2),
           m(3), m(4), m(5),
           m(6), m(7), m(8);
    // clang-format on
    // NOLINTEND(readability-magic-numbers)
    return out;
}

void CheckCorrespondences(std::span<const Eigen::Vector2d> prev,
                          std::span<const Eigen::Vector2d> curr) {
    WSLAM_ASSERT(prev.size() == curr.size(), "{} previous vs {} current points",
                 prev.size(), curr.size());
    WSLAM_ASSERT(prev.size() >= kEightPointSampleSize,
                 "{} correspondences, the eight-point fit needs {}",
                 prev.size(), kEightPointSampleSize);
}

std::vector<size_t> ClassifyInliers(const Eigen::Matrix3d& f,
                                    std::span<const Eigen::Vector2d> prev,
                                    std::span<const Eigen::Vector2d> curr,
                                    double threshold) {
    std::vector<size_t> inliers;
    inliers.reserve(prev.size());
    for (size_t i = 0; i < prev.size(); ++i) {
        if (SampsonDistance(f, prev[i], curr[i]) < threshold) {
            inliers.push_back(i);
        }
    }
    return inliers;
}

// kEightPointSampleSize distinct indices in [0, n), without allocating
std::array<size_t, kEightPointSampleSize> SampleDistinct(size_t n,
                                                         std::mt19937_64& rng) {
    WSLAM_ASSERT(n >= kEightPointSampleSize, "sampling {} of {}",
                 kEightPointSampleSize, n);

    std::uniform_int_distribution<size_t> distribution(0, n - 1);
    std::array<size_t, kEightPointSampleSize> sample{};
    size_t filled = 0;
    while (filled < sample.size()) {
        const size_t index = distribution(rng);
        const auto taken = std::span{sample}.first(filled);
        if (std::ranges::find(taken, index) == taken.end()) {
            sample.at(filled++) = index;
        }
    }
    return sample;
}

std::optional<Eigen::Matrix3d> FitSample(std::span<const Eigen::Vector2d> prev,
                                         std::span<const Eigen::Vector2d> curr,
                                         std::span<const size_t> indices) {
    std::array<Eigen::Vector2d, kEightPointSampleSize> sample_prev;
    std::array<Eigen::Vector2d, kEightPointSampleSize> sample_curr;
    for (size_t i = 0; i < indices.size(); ++i) {
        sample_prev.at(i) = prev[indices[i]];
        sample_curr.at(i) = curr[indices[i]];
    }
    return FitFundamentalMatrix(sample_prev, sample_curr);
}
}  // namespace

Eigen::Matrix3d NormalisingTransform(std::span<const Eigen::Vector2d> points) {
    WSLAM_ASSERT(!points.empty(), "no points to normalise");

    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= static_cast<double>(points.size());

    double mean_distance = 0.0;
    for (const auto& p : points) {
        mean_distance += (p - centroid).norm();
    }
    mean_distance /= static_cast<double>(points.size());

    // Coincident points: no scale to normalise. The fit on them is
    // degenerate and is rejected by its inlier count
    const double scale
        = mean_distance > 1e-12 ? std::numbers::sqrt2 / mean_distance : 1.0;

    Eigen::Matrix3d t = Eigen::Matrix3d::Identity();
    t(0, 0) = scale;
    t(1, 1) = scale;
    t(0, 2) = -scale * centroid.x();
    t(1, 2) = -scale * centroid.y();
    return t;
}

std::optional<Eigen::Matrix3d> FitFundamentalMatrix(
    std::span<const Eigen::Vector2d> prev,
    std::span<const Eigen::Vector2d> curr) {
    CheckCorrespondences(prev, curr);

    const Eigen::Matrix3d t_prev = NormalisingTransform(prev);
    const Eigen::Matrix3d t_curr = NormalisingTransform(curr);

    const Eigen::Matrix3d f_normalised
        = SolveEpipolarConstraint(prev, curr, t_prev, t_curr);

    // Rank 2: F maps every point to a line through the epipole. The
    // similarity transforms preserve rank, so it can be enforced before
    // denormalising
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        f_normalised, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd.singularValues();
    singular_values(2) = 0.0;
    const Eigen::Matrix3d f_rank2 = svd.matrixU() * singular_values.asDiagonal()
                                    * svd.matrixV().transpose();

    Eigen::Matrix3d f = t_curr.transpose() * f_rank2 * t_prev;
    if (!f.allFinite()) {
        return std::nullopt;
    }
    return f;
}

double SampsonDistance(const Eigen::Matrix3d& f, const Eigen::Vector2d& prev,
                       const Eigen::Vector2d& curr) {
    const Eigen::Vector3d xp = prev.homogeneous();
    const Eigen::Vector3d xc = curr.homogeneous();
    const Eigen::Vector3d f_xp = f * xp;
    const Eigen::Vector3d ft_xc = f.transpose() * xc;
    const double residual = xc.dot(f_xp);
    const double gradient_norm2
        = f_xp.head<2>().squaredNorm() + ft_xc.head<2>().squaredNorm();
    if (gradient_norm2 < 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    return residual * residual / gradient_norm2;
}

uint64_t RequiredRansacIterations(double inlier_ratio, double confidence,
                                  size_t sample_size) {
    WSLAM_ASSERT(inlier_ratio >= 0.0 && inlier_ratio <= 1.0, "inlier ratio {}",
                 inlier_ratio);
    WSLAM_ASSERT(confidence > 0.0 && confidence < 1.0, "confidence {}",
                 confidence);

    constexpr uint64_t kUnbounded = std::numeric_limits<uint64_t>::max();

    const double all_inlier_probability
        = std::pow(inlier_ratio, static_cast<double>(sample_size));
    if (all_inlier_probability >= 1.0) {
        return 1;
    }
    if (all_inlier_probability <= 0.0) {
        return kUnbounded;
    }

    // log1p keeps precision when the probability is tiny, where 1 - p
    // rounds to 1 and log(1 - p) to 0
    const double bound = std::ceil(std::log1p(-confidence)
                                   / std::log1p(-all_inlier_probability));
    if (!(bound < static_cast<double>(kUnbounded))) {
        return kUnbounded;
    }
    return std::max<uint64_t>(1, static_cast<uint64_t>(bound));
}

RansacOutcome EstimateFundamentalRansac(std::span<const Eigen::Vector2d> prev,
                                        std::span<const Eigen::Vector2d> curr,
                                        const RansacOptions& options,
                                        std::mt19937_64& rng) {
    WSLAM_ASSERT(prev.size() == curr.size(), "{} previous vs {} current points",
                 prev.size(), curr.size());
    WSLAM_ASSERT(options.min_inliers >= kEightPointSampleSize,
                 "min_inliers {} is below the sample size {}",
                 options.min_inliers, kEightPointSampleSize);
    WSLAM_ASSERT(options.inlier_threshold > 0.0, "inlier threshold {}",
                 options.inlier_threshold);
    WSLAM_ASSERT(options.max_iterations > 0, "no iterations allowed");

    RansacOutcome outcome;
    if (prev.size() < options.min_inliers) {
        return outcome;
    }

    std::vector<size_t> best_inliers;
    Eigen::Matrix3d best_f = Eigen::Matrix3d::Zero();
    // Unknown until a first model gives an inlier ratio
    uint64_t required = std::numeric_limits<uint64_t>::max();

    while (outcome.iterations < options.max_iterations
           && outcome.iterations < required) {
        ++outcome.iterations;

        const auto sample = SampleDistinct(prev.size(), rng);
        const auto f = FitSample(prev, curr, sample);
        if (!f) {
            ++outcome.degenerate_samples;
            continue;
        }

        auto inliers
            = ClassifyInliers(*f, prev, curr, options.inlier_threshold);
        if (inliers.size() > best_inliers.size()) {
            best_inliers = std::move(inliers);
            best_f = *f;
            required = RequiredRansacIterations(
                static_cast<double>(best_inliers.size())
                    / static_cast<double>(prev.size()),
                options.confidence);
        }
    }

    if (best_inliers.size() < options.min_inliers) {
        return outcome;
    }

    outcome.required_iterations = required;
    outcome.confidence_reached = outcome.iterations >= required;

    // Re-fit on every inlier and re-classify, so the threshold applies to
    // the refined model. Kept only when it does not lose inliers
    std::vector<Eigen::Vector2d> inlier_prev;
    std::vector<Eigen::Vector2d> inlier_curr;
    inlier_prev.reserve(best_inliers.size());
    inlier_curr.reserve(best_inliers.size());
    for (const size_t index : best_inliers) {
        inlier_prev.push_back(prev[index]);
        inlier_curr.push_back(curr[index]);
    }

    if (const auto refined = FitFundamentalMatrix(inlier_prev, inlier_curr)) {
        auto refined_inliers
            = ClassifyInliers(*refined, prev, curr, options.inlier_threshold);
        if (refined_inliers.size() >= best_inliers.size()) {
            best_f = *refined;
            best_inliers = std::move(refined_inliers);
        }
    }

    outcome.fundamental_matrix = best_f;
    outcome.inliers = std::move(best_inliers);
    outcome.model_found = true;
    return outcome;
}

std::optional<Eigen::Matrix3d> FitEssentialMatrix(
    std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr) {
    CheckCorrespondences(rays_prev, rays_curr);

    const Eigen::Matrix3d t_prev = NormalisingTransform(rays_prev);
    const Eigen::Matrix3d t_curr = NormalisingTransform(rays_curr);

    const Eigen::Matrix3d e_normalised
        = SolveEpipolarConstraint(rays_prev, rays_curr, t_prev, t_curr);
    const Eigen::Matrix3d e_raw = t_curr.transpose() * e_normalised * t_prev;
    if (!e_raw.allFinite()) {
        return std::nullopt;
    }

    // Unlike rank, equal singular values do not survive the denormalising
    // transforms, so the essential constraint is enforced on the final
    // matrix. Its scale is arbitrary; unit singular values
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        e_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d e = svd.matrixU()
                        * Eigen::Vector3d(1.0, 1.0, 0.0).asDiagonal()
                        * svd.matrixV().transpose();
    if (!e.allFinite()) {
        return std::nullopt;
    }
    return e;
}

std::array<RelativePose, 4> DecomposeEssential(const Eigen::Matrix3d& e) {
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        e, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d v = svd.matrixV();

    // E is defined up to sign, so flipping a column of U or V keeps it valid
    // and makes the recovered rotations proper (det +1)
    if (u.determinant() < 0.0) {
        u.col(2) *= -1.0;
    }
    if (v.determinant() < 0.0) {
        v.col(2) *= -1.0;
    }

    Eigen::Matrix3d w;
    w << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    const Eigen::Matrix3d r1 = u * w * v.transpose();
    const Eigen::Matrix3d r2 = u * w.transpose() * v.transpose();
    const Eigen::Vector3d t = u.col(2);

    return {
        RelativePose{.r = r1, .t = t},
        RelativePose{.r = r1, .t = -t},
        RelativePose{.r = r2, .t = t},
        RelativePose{.r = r2, .t = -t},
    };
}

std::optional<Eigen::Vector3d> TriangulatePoint(
    const RelativePose& pose, const Eigen::Vector2d& ray_prev,
    const Eigen::Vector2d& ray_curr) {
    // Calibrated rays: the previous camera is [I | 0], the current [R | t]
    Eigen::Matrix<double, 3, 4> p_prev;
    p_prev << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 4> p_curr;
    p_curr << pose.r, pose.t;

    // Two rows of x cross (P X) = 0 per view
    Eigen::Matrix4d a;
    a.row(0) = ray_prev.x() * p_prev.row(2) - p_prev.row(0);
    a.row(1) = ray_prev.y() * p_prev.row(2) - p_prev.row(1);
    a.row(2) = ray_curr.x() * p_curr.row(2) - p_curr.row(0);
    a.row(3) = ray_curr.y() * p_curr.row(2) - p_curr.row(1);

    const Eigen::JacobiSVD<Eigen::Matrix4d> svd(a, Eigen::ComputeFullV);
    const Eigen::Vector4d x = svd.matrixV().col(3);
    if (std::abs(x.w()) < 1e-12) {
        return std::nullopt;
    }

    Eigen::Vector3d point = x.head<3>() / x.w();
    if (!point.allFinite()) {
        return std::nullopt;
    }
    return point;
}

std::optional<PoseRecovery> RecoverRelativePose(
    std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr) {
    CheckCorrespondences(rays_prev, rays_curr);

    const auto e = FitEssentialMatrix(rays_prev, rays_curr);
    if (!e) {
        return std::nullopt;
    }

    std::optional<PoseRecovery> best;
    for (const RelativePose& candidate : DecomposeEssential(*e)) {
        PoseRecovery recovery;
        recovery.pose = candidate;
        recovery.points_prev.reserve(rays_prev.size());

        for (size_t i = 0; i < rays_prev.size(); ++i) {
            const auto point
                = TriangulatePoint(candidate, rays_prev[i], rays_curr[i]);
            recovery.points_prev.push_back(point);
            if (point && point->z() > 0.0
                && (candidate.r * *point + candidate.t).z() > 0.0) {
                ++recovery.in_front;
            }
        }

        // Ties keep the earlier candidate
        if (!best || recovery.in_front > best->in_front) {
            best = std::move(recovery);
        }
    }

    if (!best || best->in_front == 0) {
        return std::nullopt;
    }
    return best;
}

}  // namespace wslam::v2::util
