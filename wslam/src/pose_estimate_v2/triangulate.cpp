#include "triangulate.hpp"

#include <spdlog/spdlog.h>

#include <numbers>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "map/map_helpers_lods.hpp"
#include "models.hpp"
#include "ransac.hpp"
#include "util/camera.hpp"
#include "util/epipolar_geometry.hpp"

using namespace wslam;

#define LOG_ID "[Triangulate pass]"

v2::TriangulatePass::TriangulatePass(AnyBag& storage)
    : TriangulatePass(storage, Opts{}) {}

std::string v2::TriangulatePass::getId() const { return LOG_ID; }

std::optional<std::string> v2::TriangulatePass::initialize() {
    spdlog::info(LOG_ID
                 " Initializing (max_reproj={:.2f} px, depth=[{:.3g},{:.3g}], "
                 "min_inliers={}, refine={}, weighting={})",
                 opts_.max_reprojection_px, opts_.min_depth, opts_.max_depth,
                 opts_.min_inliers, opts_.refine,
                 static_cast<int>(opts_.weighting));

    WSLAM_ASSERT(opts_.min_inliers >= util::kEightPointSampleSize,
                 "min_inliers {} is below the eight-point sample size",
                 opts_.min_inliers);

    auto camera = util::ReadCameraParams(storage_, opts_.camera_index);
    if (!camera) {
        return std::move(camera).error();
    }
    camera_ = std::move(camera).value();

    return std::nullopt;
}

v2::KeypointTriangulationResult v2::RunTriangulation(
    const KeypointRansacResult& ransac, const data::CamSensorParams& camera,
    const TriangulatePass::Opts& opts) {
    KeypointTriangulationResult result;
    result.stats.input_matches = ransac.inliers.size();

    if (!ransac.stats.model_found || ransac.inliers.size() < opts.min_inliers) {
        return result;
    }

    std::vector<Eigen::Vector2d> rays_prev;
    std::vector<Eigen::Vector2d> rays_curr;
    rays_prev.reserve(ransac.inliers.size());
    rays_curr.reserve(ransac.inliers.size());
    for (const KeypointCorrespondence& inlier : ransac.inliers) {
        rays_prev.push_back(inlier.prev_ray);
        rays_curr.push_back(inlier.curr_ray);
    }

    auto recovery = util::RecoverRelativePose(ransac.essential_matrix,
                                              rays_prev, rays_curr);
    if (!recovery) {
        return result;
    }

    if (opts.refine) {
        const util::RefineOutcome refined = util::RefineRelativePose(
            recovery->pose, rays_prev, rays_curr,
            MatchWeights(ransac.inliers, opts.weighting), FocalLengthPx(camera),
            opts.refine_options);
        util::PoseRecovery refined_recovery
            = util::TriangulateAll(refined.pose, rays_prev, rays_curr);
        // The refinement only lowers the epipolar cost; a step that moved the
        // pose onto a cheirality flip would show up as points behind a camera
        if (refined_recovery.in_front >= recovery->in_front) {
            recovery = std::move(refined_recovery);
        }
    }

    WSLAM_ASSERT(recovery->points_prev.size() == ransac.inliers.size(),
                 "{} points for {} inliers", recovery->points_prev.size(),
                 ransac.inliers.size());

    const util::RelativePose& pose = recovery->pose;
    result.rotation = pose.r;
    result.translation = pose.t;
    result.stats.cheirality_passes = recovery->in_front;
    result.stats.rotation_angle_rad = map::util::ComputeRotationAngle(pose.r);

    const auto in_depth_range = [&](double depth) {
        return depth >= opts.min_depth && depth <= opts.max_depth;
    };

    double error_sum = 0.0;
    for (size_t i = 0; i < ransac.inliers.size(); ++i) {
        const auto& point_prev = recovery->points_prev[i];
        if (!point_prev) {
            continue;
        }
        const Eigen::Vector3d point_curr = pose.r * *point_prev + pose.t;

        if (!in_depth_range(point_prev->z())
            || !in_depth_range(point_curr.z())) {
            continue;
        }

        // Against the observed pixels, so the distortion model is part of
        // the check
        const KeypointCorrespondence& inlier = ransac.inliers[i];
        const double error
            = 0.5
              * ((util::ProjectToPixel(*point_prev, camera) - inlier.prev_px)
                     .norm()
                 + (util::ProjectToPixel(point_curr, camera) - inlier.curr_px)
                       .norm());
        // Also rejects NaN
        if (!(error <= opts.max_reprojection_px)) {
            continue;
        }

        result.landmarks.push_back(KeypointLandmark{
            .position_cam_curr = point_curr,
            .correspondence = inlier,
            .reprojection_error_px = error,
        });
        error_sum += error;
    }

    result.stats.landmark_count = result.landmarks.size();
    result.stats.pose_recovered = !result.landmarks.empty();
    if (result.stats.pose_recovered) {
        result.stats.mean_reprojection_error_px
            = error_sum / static_cast<double>(result.stats.landmark_count);
    }
    return result;
}

std::optional<std::string> v2::TriangulatePass::execute() {
    spdlog::info(LOG_ID " Executing");

    WSLAM_ASSERT(camera_.has_value(), "executing before initialize()");

    // The RANSAC pass publishes a result every frame it runs
    const auto ransac_ptr = storage_.getPtr<KeypointRansacResult>(
        ResourceIdentifier::KeypointRansacName);
    WSLAM_ASSERT(ransac_ptr.has_value(), "no RANSAC result in storage");
    const KeypointRansacResult& ransac = *ransac_ptr.value();

    KeypointTriangulationResult result
        = RunTriangulation(ransac, *camera_, opts_);

    if (!ransac.stats.model_found
        || ransac.inliers.size() < opts_.min_inliers) {
        spdlog::warn(LOG_ID
                     " RANSAC unusable (model_found={}, inliers={}, min={}); "
                     "writing an empty triangulation",
                     ransac.stats.model_found, ransac.inliers.size(),
                     opts_.min_inliers);
    } else if (result.stats.cheirality_passes == 0) {
        spdlog::warn(LOG_ID
                     " No pose: no decomposition of the essential matrix puts "
                     "a point in front of both cameras; writing an empty "
                     "triangulation");
    } else {
        constexpr double kRadToDeg = 180.0 / std::numbers::pi;
        spdlog::info(LOG_ID
                     " Done: {} landmarks (of {} inliers, {} cheirality "
                     "passes), mean reproj {:.3f} px, rotation {:.3f} deg",
                     result.stats.landmark_count, result.stats.input_matches,
                     result.stats.cheirality_passes,
                     result.stats.mean_reprojection_error_px,
                     result.stats.rotation_angle_rad * kRadToDeg);
    }

    storage_.set(ResourceIdentifier::KeypointTriangulationName,
                 std::move(result));
    return std::nullopt;
}
