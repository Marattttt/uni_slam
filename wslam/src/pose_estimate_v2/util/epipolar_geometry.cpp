#include "epipolar_geometry.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

#include "assert.hpp"

namespace wslam::v2::util {

namespace {
Eigen::Vector2d Transform(const Eigen::Matrix3d& t, const Eigen::Vector2d& p) {
    return (t * p.homogeneous()).hnormalized();
}

// Least-squares solution M of curr^T * M * prev = 0 after applying the
// normalising transforms, as the right singular vector of the stacked
// constraints with the smallest singular value. Each constraint row is
// scaled by its weight when `weights` is not empty. Returned in normalised
// space
Eigen::Matrix3d SolveEpipolarConstraint(std::span<const Eigen::Vector2d> prev,
                                        std::span<const Eigen::Vector2d> curr,
                                        std::span<const double> weights,
                                        const Eigen::Matrix3d& t_prev,
                                        const Eigen::Matrix3d& t_curr) {
    Eigen::MatrixXd a(static_cast<Eigen::Index>(prev.size()), 9);
    for (size_t i = 0; i < prev.size(); ++i) {
        const Eigen::Vector2d p = Transform(t_prev, prev[i]);
        const Eigen::Vector2d c = Transform(t_curr, curr[i]);
        const double w = weights.empty() ? 1.0 : weights[i];
        a.row(static_cast<Eigen::Index>(i)) << c.x() * p.x(), c.x() * p.y(),
            c.x(), c.y() * p.x(), c.y() * p.y(), c.y(), p.x(), p.y(), 1.0;
        a.row(static_cast<Eigen::Index>(i)) *= w;
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
                          std::span<const Eigen::Vector2d> curr,
                          std::span<const double> weights) {
    WSLAM_ASSERT(prev.size() == curr.size(), "{} previous vs {} current points",
                 prev.size(), curr.size());
    WSLAM_ASSERT(weights.empty() || weights.size() == prev.size(),
                 "{} weights for {} correspondences", weights.size(),
                 prev.size());
}

std::vector<size_t> ClassifyInliers(const Eigen::Matrix3d& e,
                                    std::span<const Eigen::Vector2d> prev,
                                    std::span<const Eigen::Vector2d> curr,
                                    double focal_px, double threshold_px2) {
    // Compare in ray units rather than scaling every distance
    const double threshold = threshold_px2 / (focal_px * focal_px);
    std::vector<size_t> inliers;
    inliers.reserve(prev.size());
    for (size_t i = 0; i < prev.size(); ++i) {
        if (SampsonDistance(e, prev[i], curr[i]) < threshold) {
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

// Least-squares essential matrix on the correspondences at `indices`
std::optional<Eigen::Matrix3d> FitSubset(std::span<const Eigen::Vector2d> prev,
                                         std::span<const Eigen::Vector2d> curr,
                                         std::span<const double> weights,
                                         std::span<const size_t> indices) {
    std::vector<Eigen::Vector2d> subset_prev;
    std::vector<Eigen::Vector2d> subset_curr;
    std::vector<double> subset_weights;
    subset_prev.reserve(indices.size());
    subset_curr.reserve(indices.size());
    subset_weights.reserve(weights.empty() ? 0 : indices.size());
    for (const size_t index : indices) {
        subset_prev.push_back(prev[index]);
        subset_curr.push_back(curr[index]);
        if (!weights.empty()) {
            subset_weights.push_back(weights[index]);
        }
    }
    return FitEssentialMatrix(subset_prev, subset_curr, subset_weights);
}

// Least-squares re-fit on the inliers, re-classified by `classify` so the
// threshold applies to the refined model, for up to `rounds` rounds while
// the inlier set changes. A re-fit whose inliers shrink was bent by a bad
// point; the fit on the previous set is kept then. On return `e` is the fit
// on `inliers` (or left as is when no fit succeeds)
void RefitOnInliers(std::span<const Eigen::Vector2d> prev,
                    std::span<const Eigen::Vector2d> curr,
                    std::span<const double> weights, const auto& classify,
                    uint32_t rounds, std::vector<size_t>& inliers,
                    Eigen::Matrix3d& e) {
    const auto refit = FitSubset(prev, curr, weights, inliers);
    if (!refit) {
        return;
    }
    e = *refit;

    for (uint32_t round = 0; round < rounds; ++round) {
        auto candidates = classify(e);
        if (candidates.size() < inliers.size() || candidates == inliers) {
            return;
        }
        const auto next = FitSubset(prev, curr, weights, candidates);
        if (!next) {
            return;
        }
        inliers = std::move(candidates);
        e = *next;
    }
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d s;
    s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return s;
}

// Signed Sampson error of one correspondence, in ray units: the algebraic
// residual over the norm of its gradient. NaN when undefined
double SignedSampsonError(const Eigen::Matrix3d& e, const Eigen::Vector2d& prev,
                          const Eigen::Vector2d& curr) {
    const Eigen::Vector3d xp = prev.homogeneous();
    const Eigen::Vector3d xc = curr.homogeneous();
    const Eigen::Vector3d e_xp = e * xp;
    const Eigen::Vector3d et_xc = e.transpose() * xc;
    const double gradient_norm2
        = e_xp.head<2>().squaredNorm() + et_xc.head<2>().squaredNorm();
    if (gradient_norm2 < 1e-24) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return xc.dot(e_xp) / std::sqrt(gradient_norm2);
}

// Pose on the 5-DoF manifold around a base pose: rotation increment
// exp(omega) applied on the left, translation moved within the tangent plane
// of the unit sphere at the base translation
struct PoseChart {
    RelativePose base;
    Eigen::Vector3d tangent_a;
    Eigen::Vector3d tangent_b;

    explicit PoseChart(RelativePose pose) : base(std::move(pose)) {
        base.t.normalize();
        // Any vector not parallel to t spans the tangent plane with it
        const Eigen::Vector3d helper = std::abs(base.t.x()) < 0.9
                                           ? Eigen::Vector3d::UnitX()
                                           : Eigen::Vector3d::UnitY();
        tangent_a = base.t.cross(helper).normalized();
        tangent_b = base.t.cross(tangent_a);
    }

    [[nodiscard]] RelativePose apply(
        const Eigen::Matrix<double, 5, 1>& delta) const {
        const Eigen::Vector3d omega = delta.head<3>();
        const double angle = omega.norm();
        const Eigen::Matrix3d rotation
            = angle > 0.0
                  ? Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix()
                  : Eigen::Matrix3d::Identity();
        return RelativePose{
            .r = rotation * base.r,
            .t = (base.t + delta(3) * tangent_a + delta(4) * tangent_b)
                     .normalized(),
        };
    }
};

double HuberCost(double residual, double delta) {
    const double magnitude = std::abs(residual);
    return magnitude <= delta ? residual * residual
                              : 2.0 * delta * magnitude - delta * delta;
}

// IRLS weight that turns the squared residual into the Huber cost
double HuberWeight(double residual, double delta) {
    const double magnitude = std::abs(residual);
    return magnitude <= delta ? 1.0 : delta / magnitude;
}

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;

// Robust weighted Sampson cost of a pose over fixed correspondences, the
// objective of RefineRelativePose
class SampsonProblem {
   public:
    SampsonProblem(std::span<const Eigen::Vector2d> rays_prev,
                   std::span<const Eigen::Vector2d> rays_curr,
                   std::span<const double> weights, double focal_px,
                   double huber_px)
        : rays_prev_(rays_prev),
          rays_curr_(rays_curr),
          weights_(weights),
          focal_px_(focal_px),
          huber_px_(huber_px) {}

    [[nodiscard]] Eigen::Index size() const {
        return static_cast<Eigen::Index>(rays_prev_.size());
    }

    // Signed Sampson errors in pixels; undefined ones contribute nothing
    [[nodiscard]] Eigen::VectorXd residuals(const RelativePose& pose) const {
        const Eigen::Matrix3d e = EssentialFromPose(pose);
        Eigen::VectorXd out(size());
        for (Eigen::Index i = 0; i < size(); ++i) {
            const auto index = static_cast<size_t>(i);
            const double error
                = SignedSampsonError(e, rays_prev_[index], rays_curr_[index]);
            out(i) = std::isnan(error) ? 0.0 : focal_px_ * error;
        }
        return out;
    }

    [[nodiscard]] double cost(const Eigen::VectorXd& residuals) const {
        double sum = 0.0;
        for (Eigen::Index i = 0; i < size(); ++i) {
            sum += weight(i) * HuberCost(residuals(i), huber_px_);
        }
        return sum;
    }

    // Gauss-Newton system (J^T W J, J^T W r) of the IRLS-weighted squares at
    // the chart's base pose, whose residuals are `residuals`. The Jacobian is
    // a central difference over the chart
    [[nodiscard]] std::pair<Matrix5d, Vector5d> normalEquations(
        const PoseChart& chart, const Eigen::VectorXd& residuals) const {
        constexpr double kStep = 1e-7;

        Eigen::MatrixXd jacobian(size(), 5);
        for (Eigen::Index k = 0; k < 5; ++k) {
            Vector5d step = Vector5d::Zero();
            step(k) = kStep;
            jacobian.col(k) = (this->residuals(chart.apply(step))
                               - this->residuals(chart.apply(-step)))
                              / (2.0 * kStep);
        }

        Eigen::VectorXd irls(size());
        for (Eigen::Index i = 0; i < size(); ++i) {
            irls(i) = weight(i) * HuberWeight(residuals(i), huber_px_);
        }
        return {jacobian.transpose() * irls.asDiagonal() * jacobian,
                jacobian.transpose() * irls.asDiagonal() * residuals};
    }

   private:
    std::span<const Eigen::Vector2d> rays_prev_;
    std::span<const Eigen::Vector2d> rays_curr_;
    std::span<const double> weights_;
    double focal_px_;
    double huber_px_;

    [[nodiscard]] double weight(Eigen::Index i) const {
        return weights_.empty() ? 1.0 : weights_[static_cast<size_t>(i)];
    }
};
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

double SampsonDistance(const Eigen::Matrix3d& m, const Eigen::Vector2d& prev,
                       const Eigen::Vector2d& curr) {
    const double error = SignedSampsonError(m, prev, curr);
    if (std::isnan(error)) {
        return std::numeric_limits<double>::infinity();
    }
    return error * error;
}

std::optional<Eigen::Matrix3d> FitEssentialMatrix(
    std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr,
    std::span<const double> weights) {
    CheckCorrespondences(rays_prev, rays_curr, weights);
    WSLAM_ASSERT(rays_prev.size() >= kEightPointSampleSize,
                 "{} correspondences, the eight-point fit needs {}",
                 rays_prev.size(), kEightPointSampleSize);

    const Eigen::Matrix3d t_prev = NormalisingTransform(rays_prev);
    const Eigen::Matrix3d t_curr = NormalisingTransform(rays_curr);

    const Eigen::Matrix3d e_normalised = SolveEpipolarConstraint(
        rays_prev, rays_curr, weights, t_prev, t_curr);
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

RansacOutcome EstimateEssentialRansac(
    std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr, std::span<const double> weights,
    double focal_px, const RansacOptions& options, std::mt19937_64& rng) {
    CheckCorrespondences(rays_prev, rays_curr, weights);
    WSLAM_ASSERT(options.min_inliers >= kEightPointSampleSize,
                 "min_inliers {} is below the sample size {}",
                 options.min_inliers, kEightPointSampleSize);
    WSLAM_ASSERT(options.inlier_threshold > 0.0, "inlier threshold {}",
                 options.inlier_threshold);
    WSLAM_ASSERT(options.max_iterations > 0, "no iterations allowed");
    WSLAM_ASSERT(focal_px > 0.0, "focal length {}", focal_px);

    RansacOutcome outcome;
    if (rays_prev.size() < options.min_inliers) {
        return outcome;
    }

    const auto classify = [&](const Eigen::Matrix3d& e) {
        return ClassifyInliers(e, rays_prev, rays_curr, focal_px,
                               options.inlier_threshold);
    };

    std::vector<size_t> best_inliers;
    Eigen::Matrix3d best_e = Eigen::Matrix3d::Zero();
    // Unknown until a first model gives an inlier ratio
    uint64_t required = std::numeric_limits<uint64_t>::max();

    while (outcome.iterations < options.max_iterations
           && outcome.iterations < required) {
        ++outcome.iterations;

        const auto sample = SampleDistinct(rays_prev.size(), rng);
        // Minimal samples are exact fits, which weights do not change
        const auto e = FitSubset(rays_prev, rays_curr, {}, sample);
        if (!e) {
            ++outcome.degenerate_samples;
            continue;
        }

        auto inliers = classify(*e);
        if (inliers.size() > best_inliers.size()) {
            best_inliers = std::move(inliers);
            best_e = *e;
            required = RequiredRansacIterations(
                static_cast<double>(best_inliers.size())
                    / static_cast<double>(rays_prev.size()),
                options.confidence);
        }
    }

    if (best_inliers.size() < options.min_inliers) {
        return outcome;
    }

    outcome.required_iterations = required;
    outcome.confidence_reached = outcome.iterations >= required;

    RefitOnInliers(rays_prev, rays_curr, weights, classify,
                   options.refit_rounds, best_inliers, best_e);

    outcome.essential_matrix = best_e;
    outcome.inliers = std::move(best_inliers);
    outcome.model_found = true;
    return outcome;
}

Eigen::Matrix3d EssentialFromPose(const RelativePose& pose) {
    return Skew(pose.t) * pose.r;
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

PoseRecovery TriangulateAll(const RelativePose& pose,
                            std::span<const Eigen::Vector2d> rays_prev,
                            std::span<const Eigen::Vector2d> rays_curr) {
    CheckCorrespondences(rays_prev, rays_curr, {});

    PoseRecovery recovery;
    recovery.pose = pose;
    recovery.points_prev.reserve(rays_prev.size());

    for (size_t i = 0; i < rays_prev.size(); ++i) {
        const auto point = TriangulatePoint(pose, rays_prev[i], rays_curr[i]);
        recovery.points_prev.push_back(point);
        if (point && point->z() > 0.0 && (pose.r * *point + pose.t).z() > 0.0) {
            ++recovery.in_front;
        }
    }
    return recovery;
}

std::optional<PoseRecovery> RecoverRelativePose(
    const Eigen::Matrix3d& e, std::span<const Eigen::Vector2d> rays_prev,
    std::span<const Eigen::Vector2d> rays_curr) {
    std::optional<PoseRecovery> best;
    for (const RelativePose& candidate : DecomposeEssential(e)) {
        PoseRecovery recovery = TriangulateAll(candidate, rays_prev, rays_curr);
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

RefineOutcome RefineRelativePose(const RelativePose& pose,
                                 std::span<const Eigen::Vector2d> rays_prev,
                                 std::span<const Eigen::Vector2d> rays_curr,
                                 std::span<const double> weights,
                                 double focal_px,
                                 const RefineOptions& options) {
    CheckCorrespondences(rays_prev, rays_curr, weights);
    WSLAM_ASSERT(focal_px > 0.0, "focal length {}", focal_px);
    WSLAM_ASSERT(options.huber_px > 0.0, "Huber threshold {}",
                 options.huber_px);

    const SampsonProblem problem(rays_prev, rays_curr, weights, focal_px,
                                 options.huber_px);

    RefineOutcome outcome;
    outcome.pose = PoseChart(pose).base;
    Eigen::VectorXd current = problem.residuals(outcome.pose);
    outcome.initial_cost = problem.cost(current);
    outcome.final_cost = outcome.initial_cost;

    // Five parameters need at least five residuals
    if (rays_prev.size() < 5) {
        return outcome;
    }

    constexpr double kMaxDamping = 1e10;
    constexpr double kMinDamping = 1e-12;
    double damping = 1e-4;

    for (uint32_t iteration = 0; iteration < options.max_iterations;
         ++iteration) {
        const PoseChart chart(outcome.pose);
        const auto [hessian, gradient]
            = problem.normalEquations(chart, current);

        // Levenberg-Marquardt: raise the damping until a step lowers the
        // cost, or give up when none does
        bool stepped = false;
        double gain = 0.0;
        while (!stepped && damping < kMaxDamping) {
            Matrix5d damped = hessian;
            damped.diagonal() += damping * hessian.diagonal().cwiseMax(1e-12);
            const Vector5d delta = damped.ldlt().solve(-gradient);
            // A non-finite step would zero every residual (undefined ones
            // count as zero), so it must not reach the cost comparison
            const RelativePose candidate
                = chart.apply(delta.allFinite() ? delta : Vector5d::Zero());
            Eigen::VectorXd candidate_residuals = problem.residuals(candidate);
            const double candidate_cost = problem.cost(candidate_residuals);
            if (delta.allFinite() && candidate_cost < outcome.final_cost) {
                gain = (outcome.final_cost - candidate_cost)
                       / std::max(outcome.final_cost, 1e-300);
                outcome.pose = candidate;
                outcome.final_cost = candidate_cost;
                current = std::move(candidate_residuals);
                stepped = true;
            } else {
                damping *= 10.0;
            }
        }

        if (!stepped) {
            break;
        }
        ++outcome.iterations;
        // Trust the quadratic model more after a successful step
        damping = std::max(damping / 10.0, kMinDamping);
        if (gain < options.relative_tolerance) {
            break;
        }
    }

    return outcome;
}

}  // namespace wslam::v2::util
