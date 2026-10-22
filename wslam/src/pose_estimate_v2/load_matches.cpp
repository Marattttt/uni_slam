#include "load_matches.hpp"

#include <spdlog/spdlog.h>

#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "compute/stage.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"
#include "map/map_helpers_lods.hpp"
#include "models.hpp"
#include "util/camera.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[Load Matches pass]"

v2::LoadMatchesPass::LoadMatchesPass(std::shared_ptr<SharedBindings> bindings,
                                     AnyBag& storage, Opts opts)
    : bindings_(std::move(bindings)), storage_(storage), opts_(opts) {
    WSLAM_ASSERT(bindings_ != nullptr, "missing shared bindings");
}

v2::LoadMatchesPass::LoadMatchesPass(std::shared_ptr<SharedBindings> bindings,
                                     AnyBag& storage)
    : LoadMatchesPass(std::move(bindings), storage, Opts{}) {}

std::string v2::LoadMatchesPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::LoadMatchesPass::initialize() {
    spdlog::info(LOG_ID " Initializing");

    auto camera = util::ReadCameraParams(storage_, opts_.camera_index);
    if (!camera) {
        return std::move(camera).error();
    }

    // The crop offsets that map model pixels to source pixels assume the
    // camera resolution is the source pyramid level's
    const auto [width, height] = camera->resolution;
    if (std::cmp_not_equal(width, model::kSourceWidth)
        || std::cmp_not_equal(height, model::kSourceHeight)) {
        return std::format(
            "camera resolution is {}x{}, the model input is cropped from {}x{}",
            width, height, model::kSourceWidth, model::kSourceHeight);
    }

    camera_ = std::move(camera).value();

    return std::nullopt;
}

std::optional<std::string> v2::LoadMatchesPass::execute() {
    // Set by the detect pass only on frames the model ran on
    if (!storage_.take<bool>(ResourceIdentifier::OrtKeypointsReadyName)
             .value_or(false)) {
        spdlog::info(LOG_ID
                     " No model output (first or skipped frame); skipping "
                     "stage");
        return compute::kStageStopExecution;
    }

    spdlog::info(LOG_ID " Executing");

    WSLAM_ASSERT(camera_.has_value(), "executing before initialize()");
    const data::CamSensorParams& camera = *camera_;

    auto matches = storage_.take<std::vector<model::KeypointMatch>>(
        ResourceIdentifier::OrtMatchesName);
    auto scores = storage_.take<std::vector<float>>(
        ResourceIdentifier::OrtMatchScoresName);
    WSLAM_ASSERT(matches.has_value() && scores.has_value(),
                 "the detect pass marked its output ready without publishing "
                 "matches and scores");
    WSLAM_ASSERT(matches->size() == scores->size(), "{} matches but {} scores",
                 matches->size(), scores->size());

    auto keypoints = storage_.take<std::vector<model::KeypointPosition>>(
        ResourceIdentifier::OrtKeypointsName);
    WSLAM_ASSERT(keypoints.has_value(),
                 "the detect pass marked its output ready without publishing "
                 "keypoints");

    size_t dropped = 0;
    size_t out_of_view = 0;
    KeypointMatches correspondences = buildCorrespondences(
        *matches, *scores, *keypoints, camera, dropped, out_of_view);

    if (dropped > 0) {
        spdlog::warn(LOG_ID
                     " Dropped {} of {} matches with a keypoint outside "
                     "the image",
                     dropped, matches->size());
    }
    if (out_of_view > 0) {
        spdlog::debug(LOG_ID " Dropped {} of {} matches with a ray past {} deg",
                      out_of_view, matches->size(), opts_.max_incidence_deg);
    }

    spdlog::info(LOG_ID " {} correspondences", correspondences.size());

    storage_.set(ResourceIdentifier::KeypointMatchesName,
                 std::move(correspondences));

    return std::nullopt;
}

v2::KeypointMatches v2::LoadMatchesPass::buildCorrespondences(
    std::span<const model::KeypointMatch> matches,
    std::span<const float> scores,
    std::span<const model::KeypointPosition> keypoints,
    const data::CamSensorParams& camera, size_t& dropped,
    size_t& out_of_view) const {
    const int64_t keypoint_count = bindings_->getKeypointCount();
    WSLAM_ASSERT(keypoints.size() == model::PairKeypointsCount(keypoint_count),
                 "{} keypoints published, expected {}", keypoints.size(),
                 model::PairKeypointsCount(keypoint_count));

    // Keypoints are stored image by image, keypoint_count each
    const auto per_image = static_cast<size_t>(keypoint_count);
    const auto prev_keypoints = keypoints.subspan(
        static_cast<size_t>(model::kLeft) * per_image, per_image);
    const auto curr_keypoints = keypoints.subspan(
        static_cast<size_t>(model::kRight) * per_image, per_image);

    const auto in_image = [](const Eigen::Vector2d& px) {
        // Pixel centres span [0, size - 1]; pixel areas reach half a pixel
        // further
        return px.allFinite() && px.x() >= -0.5 && px.y() >= -0.5
               && px.x() <= model::kSourceWidth - 0.5
               && px.y() <= model::kSourceHeight - 0.5;
    };

    const auto to_ray = [&](const Eigen::Vector2d& px) {
        return map::util::PixelToNormalised(
            px, camera, opts_.undistort_iterations, opts_.undistort_tolerance);
    };

    // A ray at the incidence limit has |ray| = tan(limit); NaN rays (past
    // 90 degrees) fail the comparison
    const double max_ray_norm
        = opts_.max_incidence_deg >= 90.0
              ? std::numeric_limits<double>::infinity()
              : std::tan(opts_.max_incidence_deg * std::numbers::pi / 180.0);
    const auto in_field_of_view = [max_ray_norm](const Eigen::Vector2d& ray) {
        return ray.allFinite() && ray.norm() <= max_ray_norm;
    };

    const auto is_keypoint_index = [keypoint_count](int64_t index) {
        return index >= 0 && index < keypoint_count;
    };

    KeypointMatches correspondences;
    correspondences.reserve(matches.size());

    for (size_t i = 0; i < matches.size(); ++i) {
        const model::KeypointMatch& match = matches[i];

        // Validated against the model's shapes by the detect pass
        WSLAM_ASSERT(match.pair_index == 0
                         && is_keypoint_index(match.left_index)
                         && is_keypoint_index(match.right_index),
                     "match row {} is [{}, {}, {}] with {} keypoints", i,
                     match.pair_index, match.left_index, match.right_index,
                     keypoint_count);

        const auto prev_index = static_cast<size_t>(match.left_index);
        const auto curr_index = static_cast<size_t>(match.right_index);

        const Eigen::Vector2d prev_px
            = model::ToSourcePixel(prev_keypoints[prev_index]);
        const Eigen::Vector2d curr_px
            = model::ToSourcePixel(curr_keypoints[curr_index]);

        if (!in_image(prev_px) || !in_image(curr_px)) {
            ++dropped;
            continue;
        }

        const Eigen::Vector2d prev_ray = to_ray(prev_px);
        const Eigen::Vector2d curr_ray = to_ray(curr_px);
        if (!in_field_of_view(prev_ray) || !in_field_of_view(curr_ray)) {
            ++out_of_view;
            continue;
        }

        correspondences.push_back(KeypointCorrespondence{
            .prev_index = static_cast<uint32_t>(prev_index),
            .curr_index = static_cast<uint32_t>(curr_index),
            .score = scores[i],
            .prev_px = prev_px,
            .curr_px = curr_px,
            .prev_ray = prev_ray,
            .curr_ray = curr_ray,
        });
    }

    return correspondences;
}
