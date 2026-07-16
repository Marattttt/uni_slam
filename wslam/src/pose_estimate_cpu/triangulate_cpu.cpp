#include "triangulate_cpu.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <numbers>
#include <vector>

#include "common.hpp"
#include "map/map_helpers_lods.hpp"
#include "models.hpp"
#include "data/provider_base.hpp"

using namespace wslam;
namespace util = wslam::map::util;

#define LOG_ID "[Triangulate pass]"

TriangulateCPU::TriangulateCPU(AnyBag& storage, Opts opts)
    : storage_(storage), opts_(opts) {}

TriangulateCPU::TriangulateCPU(AnyBag& storage)
    : TriangulateCPU(storage, Opts{}) {}

std::string TriangulateCPU::getId() const { return LOG_ID; }

namespace {
namespace impl {
// Project a 3D point in camera coordinates back to a LOD-0 pixel, applying
// the radial-tangential distortion model so that the result lives in the
// same space as the observed feature pixels.
Eigen::Vector2d ProjectToPixel(const Eigen::Vector3d& p_cam,
                               const Eigen::Vector4d& intrinsics,
                               const Eigen::Vector4d& distortion) {
    if (p_cam.z() <= 0.0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
    const Eigen::Vector2d xy(p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z());
    const Eigen::Vector2d xy_dist = util::DistortNormalised(xy, distortion);
    return {intrinsics[0] * xy_dist.x() + intrinsics[2],
            intrinsics[1] * xy_dist.y() + intrinsics[3]};
}

// Hartley-Zisserman normalising similarity for a 2D point set.
Eigen::Matrix3d NormalisingTransform(
    const std::vector<Eigen::Vector2d>& points) {
    assert(!points.empty());

    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= static_cast<double>(points.size());

    double mean_dist = 0.0;
    for (const auto& p : points) {
        mean_dist += (p - centroid).norm();
    }
    mean_dist /= static_cast<double>(points.size());

    const double scale
        = (mean_dist > 1e-12) ? (std::numbers::sqrt2 / mean_dist) : 1.0;

    Eigen::Matrix3d t = Eigen::Matrix3d::Identity();
    t(0, 0) = scale;
    t(1, 1) = scale;
    t(0, 2) = -scale * centroid.x();
    t(1, 2) = -scale * centroid.y();
    return t;
}

// Normalised eight-point algorithm tailored for the essential matrix: rank-2
// + equal non-zero singular values constraint.
std::optional<Eigen::Matrix3d> FitEssentialMatrix(
    const std::vector<Eigen::Vector2d>& prev,
    const std::vector<Eigen::Vector2d>& curr) {
    assert(prev.size() == curr.size());
    assert(prev.size() >= 8);

    const auto t_prev = NormalisingTransform(prev);
    const auto t_curr = NormalisingTransform(curr);

    const auto normalise
        = [](const Eigen::Matrix3d& t, const Eigen::Vector2d& p) {
              const Eigen::Vector3d ph(p.x(), p.y(), 1.0);
              const Eigen::Vector3d n = t * ph;
              return Eigen::Vector2d(n.x(), n.y());
          };

    Eigen::MatrixXd a(static_cast<Eigen::Index>(prev.size()), 9);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(prev.size()); ++i) {
        const auto p = normalise(t_prev, prev[static_cast<size_t>(i)]);
        const auto q = normalise(t_curr, curr[static_cast<size_t>(i)]);
        const double xp = p.x();
        const double yp = p.y();
        const double xc = q.x();
        const double yc = q.y();
        a(i, 0) = xc * xp;
        a(i, 1) = xc * yp;
        a(i, 2) = xc;
        a(i, 3) = yc * xp;
        a(i, 4) = yc * yp;
        a(i, 5) = yc;
        a(i, 6) = xp;
        a(i, 7) = yp;
        a(i, 8) = 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        a, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.matrixV().cols() < 9) {
        return std::nullopt;
    }
    const Eigen::VectorXd v = svd.matrixV().col(8);

    Eigen::Matrix3d e_norm;
    // NOLINTBEGIN(readability-magic-numbers)
    e_norm << v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8);
    // NOLINTEND(readability-magic-numbers)

    // Enforce essential-matrix shape: two equal non-zero singular values, one
    // zero singular value.
    Eigen::JacobiSVD<Eigen::Matrix3d> e_svd(
        e_norm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3d& s = e_svd.singularValues();
    const double avg = 0.5 * (s(0) + s(1));
    Eigen::Vector3d s_fixed(avg, avg, 0.0);
    const Eigen::Matrix3d e_clean
        = e_svd.matrixU() * s_fixed.asDiagonal() * e_svd.matrixV().transpose();

    Eigen::Matrix3d e_final = t_curr.transpose() * e_clean * t_prev;

    if (!e_final.allFinite()) {
        return std::nullopt;
    }
    return e_final;
}

struct PoseCandidate {
    Eigen::Matrix3d r;
    Eigen::Vector3d t;
};

// Hartley & Zisserman §9.6.2: E = U diag(1,1,0) V^T yields four (R, t)
// candidates parameterised by ±t and the two rotations R1, R2.
std::array<PoseCandidate, 4> DecomposeEssential(const Eigen::Matrix3d& e) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        e, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d v = svd.matrixV();

    // Force U and V to be in SO(3) so the recovered R has det +1.
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

    return {PoseCandidate{.r = r1, .t = t}, PoseCandidate{.r = r1, .t = -t},
            PoseCandidate{.r = r2, .t = t}, PoseCandidate{.r = r2, .t = -t}};
}

// Linear DLT triangulation: stacks the cross-product equations
//   x ~ P X
// from both views into a 4x4 system, then takes the right null vector of A.
//
// Both rays are calibrated (normalised), so the first projection is the
// identity and the second is [R | t].
std::optional<Eigen::Vector3d> TriangulatePoint(
    const Eigen::Matrix3d& r, const Eigen::Vector3d& t,
    const Eigen::Vector2d& xy_prev, const Eigen::Vector2d& xy_curr) {
    Eigen::Matrix<double, 3, 4> p1;
    p1 << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 4> p2;
    p2 << r, t;

    // NOLINTBEGIN(readability-magic-numbers)
    Eigen::Matrix4d a;
    a.row(0) = xy_prev.x() * p1.row(2) - p1.row(0);
    a.row(1) = xy_prev.y() * p1.row(2) - p1.row(1);
    a.row(2) = xy_curr.x() * p2.row(2) - p2.row(0);
    a.row(3) = xy_curr.y() * p2.row(2) - p2.row(1);
    // NOLINTEND(readability-magic-numbers)

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(
        a, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector4d x = svd.matrixV().col(3);
    if (std::abs(x.w()) < 1e-12) {
        return std::nullopt;
    }
    Eigen::Vector3d p_prev = x.head<3>() / x.w();
    if (!p_prev.allFinite()) {
        return std::nullopt;
    }
    return p_prev;
}

struct InlierCorrespondence {
    Feature feat_prev;
    Feature feat_curr;
    Eigen::Vector2d xy_prev;  // calibrated normalised image plane
    Eigen::Vector2d xy_curr;  // calibrated normalised image plane
    Eigen::Vector2d px_prev;  // LOD-0 pixel (observed, distorted)
    Eigen::Vector2d px_curr;  // LOD-0 pixel (observed, distorted)
};

std::vector<InlierCorrespondence> CollectInliers(
    const MatchResult& inliers, const data::CamSensorParams& cam,
    uint32_t undistort_iters, double undistort_tol) {
    size_t total = 0;
    for (const auto& lod_map : inliers) {
        total += lod_map.size();
    }
    std::vector<InlierCorrespondence> out;
    out.reserve(total);

    for (const auto& lod_map : inliers) {
        for (const auto& [curr, prev] : lod_map) {
            const auto px_prev = util::ToLod0Pixel(prev);
            const auto px_curr = util::ToLod0Pixel(curr);
            out.push_back(InlierCorrespondence{
                .feat_prev = prev,
                .feat_curr = curr,
                .xy_prev = util::PixelToNormalised(
                    px_prev, cam.intrinsics, cam.distortion_coefficients,
                    undistort_iters, undistort_tol),
                .xy_curr = util::PixelToNormalised(
                    px_curr, cam.intrinsics, cam.distortion_coefficients,
                    undistort_iters, undistort_tol),
                .px_prev = px_prev,
                .px_curr = px_curr,
            });
        }
    }
    return out;
}

// Counts how many correspondences land in front of both cameras under the
// given pose hypothesis. Returns triangulated 3D points (in the *previous*
// frame for now — we transform to the current frame after picking).
struct CheiralityScore {
    size_t in_front = 0;
    std::vector<Eigen::Vector3d> points_prev;
};

CheiralityScore ScoreCandidate(const PoseCandidate& c,
                               const std::vector<InlierCorrespondence>& corr) {
    CheiralityScore s;
    s.points_prev.reserve(corr.size());
    for (const auto& m : corr) {
        const auto p_opt = TriangulatePoint(c.r, c.t, m.xy_prev, m.xy_curr);
        if (!p_opt) {
            s.points_prev.emplace_back(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        const Eigen::Vector3d& p_prev = *p_opt;
        const Eigen::Vector3d p_curr = c.r * p_prev + c.t;
        s.points_prev.push_back(p_prev);
        if (p_prev.z() > 0.0 && p_curr.z() > 0.0) {
            s.in_front++;
        }
    }
    return s;
}

// Integrate gyroscope readings between two timestamps to recover the rotation
// magnitude the IMU expects. Returns the total rotation angle in radians.
//
// This is a sanity check only: we don't use it to disambiguate the pose, but
// log it side-by-side with the geometric result so unreasonable drifts can
// be spotted by eye.
double IntegrateGyroAngle(std::span<const data::IMUReading> imu) {
    if (imu.size() < 2) {
        return 0.0;
    }
    Eigen::Vector3d total = Eigen::Vector3d::Zero();
    for (size_t i = 1; i < imu.size(); ++i) {
        const auto& a = imu[i - 1];
        const auto& b = imu[i];
        const double dt_s
            = static_cast<double>(b.timestamp - a.timestamp) * 1e-9;
        if (dt_s <= 0.0) {
            continue;
        }
        const Eigen::Vector3d w_mid(
            0.5 * (static_cast<double>(a.wx()) + static_cast<double>(b.wx())),
            0.5 * (static_cast<double>(a.wy()) + static_cast<double>(b.wy())),
            0.5 * (static_cast<double>(a.wz()) + static_cast<double>(b.wz())));
        total += w_mid * dt_s;
    }
    return total.norm();
}
}  // namespace impl
}  // namespace

std::optional<std::string> TriangulateCPU::ensureIntrinsics() {
    if (intrinsics_) {
        return std::nullopt;
    }
    const auto key
        = ResourceIdentifier::GetCameraIntrinsicsName(opts_.camera_index);
    auto ptr = storage_.getPtr<data::CamSensorParams>(key);
    if (!ptr) {
        return std::format("could not get camera intrinsics under '{}'", key);
    }
    intrinsics_ = **ptr;
    spdlog::info(LOG_ID
                 " Cached intrinsics fu={:.2f} fv={:.2f} cu={:.2f} cv={:.2f}"
                 " distortion=[{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                 intrinsics_->intrinsics[0], intrinsics_->intrinsics[1],
                 intrinsics_->intrinsics[2], intrinsics_->intrinsics[3],
                 intrinsics_->distortion_coefficients[0],
                 intrinsics_->distortion_coefficients[1],
                 intrinsics_->distortion_coefficients[2],
                 intrinsics_->distortion_coefficients[3]);
    return std::nullopt;
}

std::optional<std::string> TriangulateCPU::initialize() {
    spdlog::info(LOG_ID
                 " Initializing (max_reproj={:.2f} px, depth=[{:.3g},{:.3g}], "
                 "min_inliers={}, undistort_iters={})",
                 opts_.max_reprojection_px, opts_.min_depth, opts_.max_depth,
                 opts_.min_inliers, opts_.undistort_iterations);
    return std::nullopt;
}

std::optional<std::string> TriangulateCPU::execute() {
    spdlog::info(LOG_ID " Executing");

    if (auto err = ensureIntrinsics()) {
        return err;
    }


    const auto ransac_ptr
        = storage_.getPtr<RansacResult>(ResourceIdentifier::RansacResultName);
    if (!ransac_ptr) {
        spdlog::warn(
            LOG_ID " No RansacResult in storage; writing empty triangulation");
        storage_.set(ResourceIdentifier::TriangulationResultName,
                    TriangulationResult{});
        return std::nullopt;
    }
    const auto& ransac = **ransac_ptr;

    TriangulationResult result;
    result.stats.input_matches = ransac.stats.inlier_count;

    if (!ransac.stats.model_found
        || ransac.stats.inlier_count < opts_.min_inliers) {
        spdlog::warn(LOG_ID
                     " Upstream RANSAC unusable (model_found={}, inliers={}, "
                     "min={}); writing empty triangulation",
                     ransac.stats.model_found, ransac.stats.inlier_count,
                     opts_.min_inliers);
        storage_.set(ResourceIdentifier::TriangulationResultName,
                    std::move(result));
        return std::nullopt;
    }

    const auto corr = impl::CollectInliers(ransac.inliers, *intrinsics_,
                                           opts_.undistort_iterations,
                                           opts_.undistort_tolerance);
    if (corr.size() < opts_.min_inliers) {
        spdlog::warn(LOG_ID
                     " Collected {} valid inliers (<{} min) after "
                     "undistortion; aborting",
                     corr.size(), opts_.min_inliers);
        storage_.set(ResourceIdentifier::TriangulationResultName,
                    std::move(result));
        return std::nullopt;
    }

    // Re-estimate the essential matrix from the *calibrated* rays. This is
    // cleaner than reusing K^T F K because the upstream F was fit on
    // distorted-pixel correspondences.
    std::vector<Eigen::Vector2d> rays_prev;
    std::vector<Eigen::Vector2d> rays_curr;
    rays_prev.reserve(corr.size());
    rays_curr.reserve(corr.size());
    for (const auto& m : corr) {
        rays_prev.push_back(m.xy_prev);
        rays_curr.push_back(m.xy_curr);
    }

    const auto e_opt = impl::FitEssentialMatrix(rays_prev, rays_curr);
    if (!e_opt) {
        spdlog::warn(LOG_ID
                     " Essential matrix fit failed; writing empty result");
        storage_.set(ResourceIdentifier::TriangulationResultName,
                    std::move(result));
        return std::nullopt;
    }

    const auto candidates = impl::DecomposeEssential(*e_opt);

    size_t best_idx = 0;
    impl::CheiralityScore best_score;
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto score = impl::ScoreCandidate(candidates[i], corr);
        if (score.in_front > best_score.in_front) {
            best_score = std::move(score);
            best_idx = i;
        }
    }
    const auto& best = candidates[best_idx];
    result.stats.cheirality_passes = best_score.in_front;

    if (best_score.in_front == 0) {
        spdlog::warn(LOG_ID
                     " No pose candidate has any point in front of both "
                     "cameras; writing empty result");
        storage_.set(ResourceIdentifier::TriangulationResultName,
                    std::move(result));
        return std::nullopt;
    }

    spdlog::debug(LOG_ID
                  " Selected pose candidate {} with {}/{} cheirality "
                  "passes",
                  best_idx, best_score.in_front, corr.size());

    // Build landmarks: convert points from previous-camera frame to current,
    // apply depth + reprojection filters.
    result.rotation = best.r;
    result.translation = best.t;
    result.stats.rotation_angle_rad = util::ComputeRotationAngle(best.r);

    double sum_err = 0.0;
    for (size_t i = 0; i < corr.size(); ++i) {
        const Eigen::Vector3d& p_prev = best_score.points_prev[i];
        if (!p_prev.allFinite()) {
            continue;
        }
        const Eigen::Vector3d p_curr = best.r * p_prev + best.t;

        if (p_prev.z() < opts_.min_depth || p_prev.z() > opts_.max_depth
            || p_curr.z() < opts_.min_depth || p_curr.z() > opts_.max_depth) {
            continue;
        }

        const auto reproj_prev
            = impl::ProjectToPixel(p_prev, intrinsics_->intrinsics,
                                   intrinsics_->distortion_coefficients);
        const auto reproj_curr
            = impl::ProjectToPixel(p_curr, intrinsics_->intrinsics,
                                   intrinsics_->distortion_coefficients);
        if (!reproj_prev.allFinite() || !reproj_curr.allFinite()) {
            continue;
        }
        const double err_prev = (reproj_prev - corr[i].px_prev).norm();
        const double err_curr = (reproj_curr - corr[i].px_curr).norm();
        const double err = 0.5 * (err_prev + err_curr);

        if (err > opts_.max_reprojection_px) {
            continue;
        }

        result.landmarks.push_back(Landmark{
            .position_cam_curr = p_curr,
            .feat_prev = corr[i].feat_prev,
            .feat_curr = corr[i].feat_curr,
            .reprojection_error_px = err,
        });
        sum_err += err;
    }

    result.stats.landmark_count = result.landmarks.size();
    result.stats.pose_recovered = !result.landmarks.empty();
    if (result.stats.pose_recovered) {
        result.stats.mean_reprojection_error_px
            = sum_err / static_cast<double>(result.stats.landmark_count);
    }

    // IMU sanity check — never gates the result, just a log line.
    if (const auto imu_ptr = storage_.getPtr<std::vector<data::IMUReading>>(
            ResourceIdentifier::GetImuVecName())) {
        const auto imu_angle = impl::IntegrateGyroAngle(**imu_ptr);
        constexpr double kRadToDeg = 180.0 / std::numbers::pi;
        spdlog::debug(LOG_ID
                      " Rotation: geometric={:.3f} deg, IMU-integrated "
                      "(unbiased, since startup)={:.3f} deg, imu_samples={}",
                      result.stats.rotation_angle_rad * kRadToDeg,
                      imu_angle * kRadToDeg, (*imu_ptr)->size());
    }

    spdlog::info(LOG_ID
                 " Triangulation done: {} landmarks (of {} inliers, "
                 "{} cheirality passes), mean reproj {:.3f} px, "
                 "rotation {:.3f} deg",
                 result.stats.landmark_count, result.stats.input_matches,
                 result.stats.cheirality_passes,
                 result.stats.mean_reprojection_error_px,
                 result.stats.rotation_angle_rad * (180.0 / std::numbers::pi));

    storage_.set(ResourceIdentifier::TriangulationResultName, std::move(result));
    return std::nullopt;
}
