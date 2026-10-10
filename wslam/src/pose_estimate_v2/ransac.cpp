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

std::optional<std::string> v2::RansacPass::initialize() {
    const util::RansacOptions& ransac = opts_.ransac;
    spdlog::info(LOG_ID
                 " Initializing (confidence={}, inlier_thresh={:.3f} px^2, "
                 "min_inliers={}, max_iters={}, seed=0x{:x})",
                 ransac.confidence, ransac.inlier_threshold, ransac.min_inliers,
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

    KeypointRansacResult result;
    result.stats.total_matches = matches.size();

    std::vector<Eigen::Vector2d> prev;
    std::vector<Eigen::Vector2d> curr;
    prev.reserve(matches.size());
    curr.reserve(matches.size());
    for (const KeypointCorrespondence& match : matches) {
        prev.push_back(util::NormalisedToPixel(match.prev_ray, *camera_));
        curr.push_back(util::NormalisedToPixel(match.curr_ray, *camera_));
    }

    const util::RansacOutcome outcome
        = util::EstimateFundamentalRansac(prev, curr, opts_.ransac, rng_);

    result.stats.iterations_run = outcome.iterations;
    result.required_iterations = outcome.required_iterations;
    result.confidence_reached = outcome.confidence_reached;

    if (outcome.degenerate_samples > 0) {
        spdlog::debug(LOG_ID " {} of {} samples were degenerate",
                      outcome.degenerate_samples, outcome.iterations);
    }

    if (!outcome.model_found) {
        spdlog::warn(LOG_ID
                     " No consensus model ({} matches, {} iterations, min {} "
                     "inliers); writing an empty inlier set",
                     matches.size(), outcome.iterations,
                     opts_.ransac.min_inliers);
        storage_.set(ResourceIdentifier::KeypointRansacName, std::move(result));
        return std::nullopt;
    }

    if (!outcome.confidence_reached) {
        spdlog::warn(LOG_ID
                     " Stopped at the {}-iteration guard; the {} confidence "
                     "needs {} at this inlier ratio",
                     outcome.iterations, opts_.ransac.confidence,
                     outcome.required_iterations);
    }

    result.inliers.reserve(outcome.inliers.size());
    for (const size_t index : outcome.inliers) {
        result.inliers.push_back(matches.at(index));
    }
    result.fundamental_matrix = outcome.fundamental_matrix;
    result.stats.inlier_count = outcome.inliers.size();
    result.stats.model_found = true;

    spdlog::info(LOG_ID
                 " Done: {}/{} inliers ({:.1f}%), {} iterations (confidence "
                 "needs {})",
                 result.stats.inlier_count, result.stats.total_matches,
                 100.0 * static_cast<double>(result.stats.inlier_count)
                     / static_cast<double>(
                         std::max<size_t>(1, result.stats.total_matches)),
                 result.stats.iterations_run, result.required_iterations);

    storage_.set(ResourceIdentifier::KeypointRansacName, std::move(result));

    return std::nullopt;
}
