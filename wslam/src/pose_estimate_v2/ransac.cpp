#include "ransac.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "models.hpp"
#include "util/camera.hpp"

using namespace wslam;

#define LOG_ID "[RANSAC pass]"

v2::RansacPass::RansacPass(AnyBag& storage) : RansacPass(storage, Opts{}) {}

std::string v2::RansacPass::getId() const { return LOG_ID; }

double v2::FocalLengthPx(const data::CamSensorParams& camera) {
    return 0.5 * (camera.intrinsics[0] + camera.intrinsics[1]);
}

v2::KeypointRansacResult v2::RunRansac(const KeypointMatches& matches,
                                       const data::CamSensorParams& camera,
                                       const RansacPass::Opts& opts,
                                       std::mt19937_64& rng) {
    KeypointRansacResult result;
    result.stats.total_matches = matches.size();

    const KeypointMatches filtered = FilterMatchesByScore(matches, opts.filter);
    result.filtered_matches = filtered.size();

    std::vector<Eigen::Vector2d> rays_prev;
    std::vector<Eigen::Vector2d> rays_curr;
    rays_prev.reserve(filtered.size());
    rays_curr.reserve(filtered.size());
    for (const KeypointCorrespondence& match : filtered) {
        rays_prev.push_back(match.prev_ray);
        rays_curr.push_back(match.curr_ray);
    }
    const std::vector<double> weights = MatchWeights(filtered, opts.weighting);

    const util::RansacOutcome outcome = util::EstimateEssentialRansac(
        rays_prev, rays_curr, weights, FocalLengthPx(camera), opts.ransac, rng);

    result.stats.iterations_run = outcome.iterations;
    result.required_iterations = outcome.required_iterations;
    result.confidence_reached = outcome.confidence_reached;

    if (outcome.degenerate_samples > 0) {
        spdlog::debug(LOG_ID " {} of {} samples were degenerate",
                      outcome.degenerate_samples, outcome.iterations);
    }

    if (!outcome.model_found) {
        return result;
    }

    result.inliers.reserve(outcome.inliers.size());
    for (const size_t index : outcome.inliers) {
        result.inliers.push_back(filtered.at(index));
    }
    result.essential_matrix = outcome.essential_matrix;
    result.stats.inlier_count = outcome.inliers.size();
    result.stats.model_found = true;
    return result;
}

std::optional<std::string> v2::RansacPass::initialize() {
    const util::RansacOptions& ransac = opts_.ransac;
    spdlog::info(LOG_ID
                 " Initializing (min_score={:.2f}, min_matches={}, "
                 "weighting={}, confidence={}, inlier_thresh={:.3f} px^2, "
                 "min_inliers={}, max_iters={}, seed=0x{:x})",
                 opts_.filter.min_score, opts_.filter.min_matches,
                 static_cast<int>(opts_.weighting), ransac.confidence,
                 ransac.inlier_threshold, ransac.min_inliers,
                 ransac.max_iterations, opts_.rng_seed);

    auto camera = util::ReadCameraParams(storage_, opts_.camera_index);
    if (!camera) {
        return std::move(camera).error();
    }
    camera_ = std::move(camera).value();

    return std::nullopt;
}

std::optional<std::string> v2::RansacPass::execute() {
    spdlog::info(LOG_ID " Executing");

    WSLAM_ASSERT(camera_.has_value(), "executing before initialize()");

    // The load pass publishes the matches or stops the stage before this
    const auto matches_ptr = storage_.getPtr<KeypointMatches>(
        ResourceIdentifier::KeypointMatchesName);
    WSLAM_ASSERT(matches_ptr.has_value(), "no keypoint matches in storage");
    const KeypointMatches& matches = *matches_ptr.value();

    KeypointRansacResult result = RunRansac(matches, *camera_, opts_, rng_);

    if (!result.stats.model_found) {
        spdlog::warn(LOG_ID
                     " No consensus model ({} matches, {} after the score "
                     "filter, {} iterations, min {} inliers); writing an "
                     "empty inlier set",
                     matches.size(), result.filtered_matches,
                     result.stats.iterations_run, opts_.ransac.min_inliers);
    } else {
        if (!result.confidence_reached) {
            spdlog::warn(LOG_ID
                         " Stopped at the {}-iteration guard; the {} "
                         "confidence needs {} at this inlier ratio",
                         result.stats.iterations_run, opts_.ransac.confidence,
                         result.required_iterations);
        }

        spdlog::info(
            LOG_ID
            " Done: {}/{} inliers ({:.1f}%) of {} matches, {} iterations "
            "(confidence needs {})",
            result.stats.inlier_count, result.filtered_matches,
            100.0 * static_cast<double>(result.stats.inlier_count)
                / static_cast<double>(
                    std::max<size_t>(1, result.filtered_matches)),
            result.stats.total_matches, result.stats.iterations_run,
            result.required_iterations);
    }

    storage_.set(ResourceIdentifier::KeypointRansacName, std::move(result));

    return std::nullopt;
}
