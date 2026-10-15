#include "pnp_verify.hpp"

#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;

namespace {
// Correspondences a linear (DLT) pose needs: 12 unknowns, 2 equations each
constexpr size_t kSampleSize = 6;

// Gauss-Newton refinement of the RANSAC pose on its inliers
constexpr uint32_t kRefineIterations = 10;
constexpr double kRefineStepTolerance = 1e-10;

// Fixed, so a run's verification decisions are reproducible
constexpr uint64_t kRansacSeed = 0x5eed'1001;

using Correspondences = std::span<const PnpCorrespondence>;

// Pose as the camera sees the points: p_query = R * p_candidate + t
struct CameraFromPoints {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// Squared reprojection error on the normalised image plane; infinite behind
// the camera
[[nodiscard]] double SquaredError(const CameraFromPoints& pose,
                                  const PnpCorrespondence& c) {
    const Eigen::Vector3d p = pose.R * c.point_candidate + pose.t;
    if (p.z() <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return (p.head<2>() / p.z() - c.query_ray).squaredNorm();
}

// Linear pose from >= 6 correspondences: the projection matrix P = [M | t]
// by DLT on Hartley-normalised points, then the nearest rotation to M
[[nodiscard]] std::optional<CameraFromPoints> SolveDlt(
    Correspondences all, std::span<const size_t> indices) {
    const auto n = static_cast<Eigen::Index>(indices.size());

    // Condition the points: centroid at the origin, mean distance sqrt(3)
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const size_t i : indices) {
        centroid += all[i].point_candidate;
    }
    centroid /= static_cast<double>(n);
    double mean_distance = 0.0;
    for (const size_t i : indices) {
        mean_distance += (all[i].point_candidate - centroid).norm();
    }
    mean_distance /= static_cast<double>(n);
    if (mean_distance < 1e-9) {
        return std::nullopt;
    }
    const double scale = std::numbers::sqrt3 / mean_distance;

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(2 * n, 12);
    for (Eigen::Index row = 0; row < n; ++row) {
        const PnpCorrespondence& c = all[indices[static_cast<size_t>(row)]];
        Eigen::Vector4d x;
        x << scale * (c.point_candidate - centroid), 1.0;
        const double u = c.query_ray.x();
        const double v = c.query_ray.y();
        a.block<1, 4>(2 * row, 0) = x.transpose();
        a.block<1, 4>(2 * row, 8) = -u * x.transpose();
        a.block<1, 4>(2 * row + 1, 4) = x.transpose();
        a.block<1, 4>(2 * row + 1, 8) = -v * x.transpose();
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
    const Eigen::VectorXd p = svd.matrixV().col(11);

    // P' acts on the conditioned points X' = scale * (X - centroid)
    Eigen::Matrix<double, 3, 4> conditioned;
    conditioned.row(0) = p.segment<4>(0).transpose();
    conditioned.row(1) = p.segment<4>(4).transpose();
    conditioned.row(2) = p.segment<4>(8).transpose();

    Eigen::Matrix3d m = conditioned.leftCols<3>() * scale;
    Eigen::Vector3d t = conditioned.col(3) - m * centroid;

    // The null vector's sign is arbitrary: M must be a positive multiple of
    // a rotation
    if (m.determinant() < 0.0) {
        m = -m;
        t = -t;
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> rotation_svd(
        m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const double mean_singular = rotation_svd.singularValues().mean();
    if (mean_singular < 1e-12) {
        return std::nullopt;
    }

    CameraFromPoints pose;
    pose.R = rotation_svd.matrixU() * rotation_svd.matrixV().transpose();
    if (pose.R.determinant() < 0.0) {
        return std::nullopt;
    }
    pose.t = t / mean_singular;
    return pose;
}

[[nodiscard]] std::vector<size_t> Inliers(const CameraFromPoints& pose,
                                          Correspondences all,
                                          double threshold_sq) {
    std::vector<size_t> inliers;
    for (size_t i = 0; i < all.size(); ++i) {
        if (SquaredError(pose, all[i]) < threshold_sq) {
            inliers.push_back(i);
        }
    }
    return inliers;
}

// Huber-weighted Gauss-Newton on the camera pose (query in the candidate
// frame), over the given correspondences
[[nodiscard]] gtsam::Pose3 Refine(gtsam::Pose3 camera_pose,
                                  Correspondences all,
                                  std::span<const size_t> indices,
                                  double huber_threshold) {
    for (uint32_t iteration = 0; iteration < kRefineIterations; ++iteration) {
        gtsam::Matrix66 hessian = gtsam::Matrix66::Zero();
        gtsam::Vector6 gradient = gtsam::Vector6::Zero();

        const gtsam::PinholeBase camera{camera_pose};
        for (const size_t i : indices) {
            gtsam::Matrix26 jacobian;
            gtsam::Point2 projected;
            try {
                projected = camera.project2(all[i].point_candidate, jacobian);
            } catch (const gtsam::CheiralityException&) {
                continue;
            }
            const gtsam::Vector2 residual = projected - all[i].query_ray;
            const double norm = residual.norm();
            const double weight
                = norm <= huber_threshold ? 1.0 : huber_threshold / norm;
            hessian += weight * jacobian.transpose() * jacobian;
            gradient += weight * jacobian.transpose() * residual;
        }

        const gtsam::Vector6 step = hessian.ldlt().solve(-gradient);
        if (!step.allFinite()) {
            break;
        }
        camera_pose = camera_pose.retract(step);
        if (step.squaredNorm() < kRefineStepTolerance) {
            break;
        }
    }
    return camera_pose;
}

[[nodiscard]] gtsam::Pose3 ToCameraPose(const CameraFromPoints& pose) {
    // p_query = R p_candidate + t  =>  camera pose R^T, -R^T t
    return gtsam::Pose3{gtsam::Rot3{pose.R.transpose()},
                        gtsam::Point3{-pose.R.transpose() * pose.t}};
}

[[nodiscard]] CameraFromPoints FromCameraPose(const gtsam::Pose3& camera_pose) {
    const gtsam::Pose3 inverse = camera_pose.inverse();
    return {.R = inverse.rotation().matrix(), .t = inverse.translation()};
}
}  // namespace

std::expected<PnpResult, std::string> loop_closure::SolvePnp(
    std::span<const PnpCorrespondence> correspondences, double focal_px,
    const MatchAcceptance& acceptance) {
    const size_t n = correspondences.size();
    if (n < std::max<size_t>(acceptance.min_correspondences, kSampleSize)) {
        return std::unexpected(std::format("{} 2D-3D correspondences < {}", n,
                                           acceptance.min_correspondences));
    }

    // Threshold on the normalised image plane
    const double threshold = acceptance.pnp_reprojection_px / focal_px;
    const double threshold_sq = threshold * threshold;

    std::mt19937_64 rng(kRansacSeed);  // NOLINT(bugprone-random-generator-seed)
    std::vector<size_t> pool(n);
    std::ranges::iota(pool, size_t{0});

    std::optional<CameraFromPoints> best;
    std::vector<size_t> best_inliers;
    uint64_t required = acceptance.pnp_max_iterations;

    for (uint64_t iteration = 0;
         iteration < std::min<uint64_t>(required, acceptance.pnp_max_iterations);
         ++iteration) {
        // Partial Fisher-Yates: the first kSampleSize entries are the sample
        for (size_t i = 0; i < kSampleSize; ++i) {
            std::uniform_int_distribution<size_t> pick(i, n - 1);
            std::swap(pool[i], pool[pick(rng)]);
        }
        const auto hypothesis
            = SolveDlt(correspondences, std::span{pool}.first(kSampleSize));
        if (!hypothesis) {
            continue;
        }

        auto inliers = Inliers(*hypothesis, correspondences, threshold_sq);
        if (inliers.size() <= best_inliers.size()) {
            continue;
        }
        best = hypothesis;
        best_inliers = std::move(inliers);

        // N = log(1 - p) / log(1 - w^s) for the best inlier ratio w
        const double ratio = static_cast<double>(best_inliers.size())
                             / static_cast<double>(n);
        const double all_inlier_sample
            = std::pow(ratio, static_cast<double>(kSampleSize));
        if (all_inlier_sample >= 1.0 - 1e-12) {
            required = iteration + 1;
        } else if (all_inlier_sample > 0.0) {
            required = static_cast<uint64_t>(std::ceil(
                std::log(1.0 - acceptance.pnp_confidence)
                / std::log(1.0 - all_inlier_sample)));
        }
    }

    if (!best || best_inliers.size() < kSampleSize) {
        return std::unexpected("PnP RANSAC found no pose");
    }

    // Re-fit linearly on all inliers, refine, then re-classify
    CameraFromPoints fitted
        = SolveDlt(correspondences, best_inliers).value_or(*best);
    gtsam::Pose3 camera_pose
        = Refine(ToCameraPose(fitted), correspondences,
                 Inliers(fitted, correspondences, threshold_sq), threshold);
    std::vector<size_t> inliers
        = Inliers(FromCameraPose(camera_pose), correspondences, threshold_sq);
    if (inliers.size() < best_inliers.size()) {
        // The refinement drifted; keep the RANSAC pose
        camera_pose = Refine(ToCameraPose(*best), correspondences,
                             best_inliers, threshold);
        inliers = Inliers(FromCameraPose(camera_pose), correspondences,
                          threshold_sq);
    }

    const double inlier_ratio
        = static_cast<double>(inliers.size()) / static_cast<double>(n);
    if (inliers.size() < acceptance.min_pnp_inliers
        || inlier_ratio < acceptance.min_inlier_ratio) {
        return std::unexpected(
            std::format("{} PnP inliers of {} ({:.2f}), need {} and {:.2f}",
                        inliers.size(), n, inlier_ratio,
                        acceptance.min_pnp_inliers,
                        acceptance.min_inlier_ratio));
    }

    return PnpResult{
        .T_candidate_query = camera_pose,
        .inliers = inliers.size(),
        .correspondences = n,
        .inlier_indices = std::move(inliers),
    };
}
