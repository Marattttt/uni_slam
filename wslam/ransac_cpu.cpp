#include "ransac_cpu.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <numbers>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[RANSAC Features pass]"

RansacCPU::RansacCPU(GpuSharedBindings& shared,
                     std::shared_ptr<compute::GPU> gpu)
    : RansacCPU(shared, std::move(gpu), Opts{}) {}

std::string RansacCPU::getId() const { return LOG_ID; }

namespace {
namespace impl {

// Internal flat representation of a match used during fitting. The lod index
// is retained so the filtered output can be regrouped per-LOD.
struct FlatMatch {
    Eigen::Vector2d prev;  // LOD-0 pixel coords
    Eigen::Vector2d curr;  // LOD-0 pixel coords
    Feature feat_prev;
    Feature feat_curr;
};

constexpr double LodScale(uint32_t lod) {
    double scale = 1.0;
    for (uint32_t i = 0; i < lod; ++i) {
        scale *= GPUConst::lod_scale_factor;
    }
    return scale;
}

constexpr Eigen::Vector2d ToLod0(const Feature& f) {
    const double scale = LodScale(f.lod);
    return {static_cast<double>(f.x) * scale,
            static_cast<double>(f.y) * scale};
}

std::vector<FlatMatch> FlattenMatches(const MatchResult& matches) {
    std::vector<FlatMatch> out;
    out.reserve(std::ranges::fold_left(matches, 0UZ,
                                       [](size_t acc, const auto& lod_map) {
                                           return acc + lod_map.size();
                                       }));

    // MatchResult stores {curr_feature → prev_feature}; the per-LOD index is
    // the LOD of the *current* feature (key). The previous feature may live
    // on an adjacent LOD, which is fine — we always project to LOD-0 for the
    // geometric fit.
    for (const auto& lod_map : matches) {
        for (const auto& [curr, prev] : lod_map) {
            out.push_back(FlatMatch{
                .prev = ToLod0(prev),
                .curr = ToLod0(curr),
                .feat_prev = prev,
                .feat_curr = curr,
            });
        }
    }
    return out;
}

// Computes the Hartley-Zisserman normalising similarity transform for a 2D
// point set: centroid → origin, mean distance from origin → sqrt(2).
//
// Returns the 3x3 transform T such that x_norm_h = T * x_h (homogeneous).
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

    // Degenerate input (all points coincident) — return identity to avoid
    // dividing by zero. The caller will discard the resulting F anyway since
    // it cannot produce inliers.
    const double scale = (mean_dist > 1e-12) ? (std::numbers::sqrt2 / mean_dist)
                                             : 1.0;

    Eigen::Matrix3d t = Eigen::Matrix3d::Identity();
    t(0, 0) = scale;
    t(1, 1) = scale;
    t(0, 2) = -scale * centroid.x();
    t(1, 2) = -scale * centroid.y();
    return t;
}

// Normalised 8-point algorithm. Accepts any sample size >= 8 (re-fitting on
// the full inlier set uses the same routine). Returns std::nullopt when the
// linear system is degenerate.
std::optional<Eigen::Matrix3d> FitFundamentalMatrix(
    const std::vector<Eigen::Vector2d>& prev,
    const std::vector<Eigen::Vector2d>& curr) {
    assert(prev.size() == curr.size());
    assert(prev.size() >= 8);

    const auto t_prev = NormalisingTransform(prev);
    const auto t_curr = NormalisingTransform(curr);

    const auto normalise = [](const Eigen::Matrix3d& t,
                              const Eigen::Vector2d& p) {
        const Eigen::Vector3d ph(p.x(), p.y(), 1.0);
        const Eigen::Vector3d n = t * ph;
        return Eigen::Vector2d(n.x(), n.y());
    };

    Eigen::MatrixXd a(prev.size(), 9);
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
    const Eigen::VectorXd f = svd.matrixV().col(8);

    Eigen::Matrix3d f_norm;
    // Row-major reshape of the 9-vector into a 3x3 matrix.
    // NOLINTBEGIN(readability-magic-numbers)
    f_norm << f(0), f(1), f(2),
              f(3), f(4), f(5),
              f(6), f(7), f(8);
    // NOLINTEND(readability-magic-numbers)

    // Enforce rank 2 by zeroing the smallest singular value. The fundamental
    // matrix has a non-trivial null space (the epipoles).
    Eigen::JacobiSVD<Eigen::Matrix3d> f_svd(
        f_norm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = f_svd.singularValues();
    s(2) = 0.0;
    const Eigen::Matrix3d f_rank2
        = f_svd.matrixU() * s.asDiagonal() * f_svd.matrixV().transpose();

    // Denormalise.
    Eigen::Matrix3d f_final = t_curr.transpose() * f_rank2 * t_prev;

    if (!f_final.allFinite()) {
        return std::nullopt;
    }
    return f_final;
}

// Sampson distance squared between a correspondence (prev, curr) and the
// epipolar geometry encoded by F. Treats `prev` as the point in the first
// image (column of F) and `curr` as the point in the second (row of F),
// matching the convention used when assembling the linear system above.
double SampsonDistance(const Eigen::Matrix3d& f, const Eigen::Vector2d& prev,
                       const Eigen::Vector2d& curr) {
    const Eigen::Vector3d xp(prev.x(), prev.y(), 1.0);
    const Eigen::Vector3d xc(curr.x(), curr.y(), 1.0);
    const Eigen::Vector3d fxp = f * xp;
    const Eigen::Vector3d ftxc = f.transpose() * xc;
    const double num = xc.dot(fxp);
    const double denom = fxp.x() * fxp.x() + fxp.y() * fxp.y()
                         + ftxc.x() * ftxc.x() + ftxc.y() * ftxc.y();
    if (denom < 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    return (num * num) / denom;
}

std::vector<size_t> ClassifyInliers(const Eigen::Matrix3d& f,
                                    const std::vector<FlatMatch>& matches,
                                    double threshold) {
    std::vector<size_t> inliers;
    inliers.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        const double d = SampsonDistance(f, matches[i].prev, matches[i].curr);
        if (d < threshold) {
            inliers.push_back(i);
        }
    }
    return inliers;
}

// Picks `k` distinct indices in [0, n) without replacement.
std::array<size_t, 8> SampleEight(size_t n, std::mt19937_64& rng) {
    assert(n >= 8);
    std::array<size_t, 8> out{};
    std::unordered_set<size_t> picked;
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t filled = 0;
    while (filled < out.size()) {
        const size_t s = dist(rng);
        if (picked.insert(s).second) {
            out[filled++] = s;
        }
    }
    return out;
}

MatchResult BuildFilteredMatchResult(const std::vector<FlatMatch>& matches,
                                     const std::vector<size_t>& inliers) {
    MatchResult result;
    for (const auto idx : inliers) {
        const auto& m = matches[idx];
        // Original MatchResult keys by current-frame feature (per its LOD).
        const auto lod = m.feat_curr.lod;
        assert(lod < GPUConst::levels_of_detail);
        result[lod].emplace(m.feat_curr, m.feat_prev);
    }
    return result;
}

std::array<std::array<double, 3>, 3> ToPlainMatrix(const Eigen::Matrix3d& f) {
    std::array<std::array<double, 3>, 3> out{};
    for (Eigen::Index r = 0; r < 3; ++r) {
        for (Eigen::Index c = 0; c < 3; ++c) {
            out[static_cast<size_t>(r)][static_cast<size_t>(c)] = f(r, c);
        }
    }
    return out;
}

}  // namespace impl
}  // namespace

std::optional<std::string> RansacCPU::initialize() {
    spdlog::info(LOG_ID
                 " Initializing (max_iters={}, inlier_thresh={:.3f} px^2, "
                 "min_matches={}, seed=0x{:x})",
                 opts_.max_iterations, opts_.inlier_threshold,
                 opts_.min_matches, opts_.rng_seed);
    return std::nullopt;
}

std::optional<std::string> RansacCPU::execute() {
    spdlog::info(LOG_ID " Executing");

    auto& storage = shared_.getStorage();

    const auto matches_ptr
        = storage.getPtr<MatchResult>(ResourceIdentifier::MatchedFeaturesName);

    if (!matches_ptr) {
        spdlog::warn(LOG_ID
                     " No MatchResult found in storage; skipping RANSAC for "
                     "this frame");
        storage.set(ResourceIdentifier::RansacResultName, RansacResult{});
        return std::nullopt;
    }

    const auto flat = impl::FlattenMatches(**matches_ptr);

    RansacResult result;
    result.stats.total_matches = flat.size();

    if (flat.size() < opts_.min_matches) {
        spdlog::warn(LOG_ID
                     " Too few matches for geometric fit ({} < {}); writing "
                     "empty inlier set",
                     flat.size(), opts_.min_matches);
        storage.set(ResourceIdentifier::RansacResultName, std::move(result));
        return std::nullopt;
    }

    std::vector<size_t> best_inliers;
    Eigen::Matrix3d best_f = Eigen::Matrix3d::Zero();

    uint32_t iter = 0;
    uint32_t degenerate_samples = 0;
    for (; iter < opts_.max_iterations; ++iter) {
        const auto sample = impl::SampleEight(flat.size(), rng_);

        std::vector<Eigen::Vector2d> sample_prev(8);
        std::vector<Eigen::Vector2d> sample_curr(8);
        for (size_t i = 0; i < 8; ++i) {
            sample_prev[i] = flat[sample[i]].prev;
            sample_curr[i] = flat[sample[i]].curr;
        }

        const auto f_opt = impl::FitFundamentalMatrix(sample_prev, sample_curr);
        if (!f_opt) {
            degenerate_samples++;
            continue;
        }

        auto inliers = impl::ClassifyInliers(*f_opt, flat,
                                             opts_.inlier_threshold);
        if (inliers.size() > best_inliers.size()) {
            spdlog::debug(LOG_ID
                          " iter {}/{}: new best with {} inliers (was {})",
                          iter, opts_.max_iterations, inliers.size(),
                          best_inliers.size());
            best_inliers = std::move(inliers);
            best_f = *f_opt;
        }
    }

    result.stats.iterations_run = iter;

    if (degenerate_samples > 0) {
        spdlog::debug(LOG_ID " {} of {} random samples were degenerate",
                      degenerate_samples, iter);
    }

    if (best_inliers.size() < opts_.min_matches) {
        spdlog::warn(LOG_ID
                     " RANSAC failed to find a consensus set "
                     "(best={} inliers out of {} matches, {} iterations)",
                     best_inliers.size(), flat.size(), iter);
        storage.set(ResourceIdentifier::RansacResultName, std::move(result));
        return std::nullopt;
    }

    // Final re-fit on the full inlier set, then re-classify so the threshold
    // is applied consistently to the refined model.
    std::vector<Eigen::Vector2d> refit_prev;
    std::vector<Eigen::Vector2d> refit_curr;
    refit_prev.reserve(best_inliers.size());
    refit_curr.reserve(best_inliers.size());
    for (const auto idx : best_inliers) {
        refit_prev.push_back(flat[idx].prev);
        refit_curr.push_back(flat[idx].curr);
    }

    if (auto f_refined = impl::FitFundamentalMatrix(refit_prev, refit_curr)) {
        auto refined_inliers = impl::ClassifyInliers(*f_refined, flat,
                                                    opts_.inlier_threshold);
        if (refined_inliers.size() >= best_inliers.size()) {
            best_f = *f_refined;
            best_inliers = std::move(refined_inliers);
            spdlog::debug(LOG_ID " refit improved inlier count to {}",
                          best_inliers.size());
        } else {
            spdlog::debug(LOG_ID
                          " refit was worse ({} vs {}); keeping RANSAC model",
                          refined_inliers.size(), best_inliers.size());
        }
    } else {
        spdlog::warn(LOG_ID " final refit failed; keeping RANSAC model");
    }

    result.inliers = impl::BuildFilteredMatchResult(flat, best_inliers);
    result.fundamental_matrix = impl::ToPlainMatrix(best_f);
    result.stats.inlier_count = best_inliers.size();
    result.stats.model_found = true;

    spdlog::info(LOG_ID
                 " RANSAC done: {}/{} inliers ({:.1f}%), {} iterations, "
                 "threshold {:.2f} px^2",
                 result.stats.inlier_count, result.stats.total_matches,
                 100.0 * static_cast<double>(result.stats.inlier_count)
                     / static_cast<double>(std::max<size_t>(
                         1, result.stats.total_matches)),
                 result.stats.iterations_run, opts_.inlier_threshold);

    storage.set(ResourceIdentifier::RansacResultName, std::move(result));

    return std::nullopt;
}
