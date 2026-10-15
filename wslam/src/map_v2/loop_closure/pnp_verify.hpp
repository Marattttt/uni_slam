#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <cstddef>
#include <expected>
#include <span>
#include <string>

#include "config.hpp"

namespace wslam::v2::mapping::loop_closure {

// A landmark of the candidate keyframe seen by the query keyframe
struct PnpCorrespondence {
    // Landmark in the candidate camera frame, metres
    Eigen::Vector3d point_candidate;
    // Undistorted normalised image coordinates of its query observation
    Eigen::Vector2d query_ray;
};

struct PnpResult {
    // Pose of the query camera in the candidate camera frame:
    // p_candidate = T_candidate_query * p_query
    gtsam::Pose3 T_candidate_query;
    size_t inliers = 0;
    size_t correspondences = 0;
};

// RANSAC PnP on calibrated rays: 6-point linear (DLT) hypotheses, an
// adaptive sample count for the acceptance's confidence, then a linear re-fit
// and Huber Gauss-Newton refinement on the inliers. The pixel threshold is
// scaled to the normalised image plane by focal_px. Fails, with the reason,
// when the acceptance's inlier gates reject the pose
//
// Deliberately no minimal (P3P) solver: with tens to hundreds of
// correspondences the extra samples DLT needs are cheap, and it is far
// simpler to get right
[[nodiscard]] std::expected<PnpResult, std::string> SolvePnp(
    std::span<const PnpCorrespondence> correspondences, double focal_px,
    const MatchAcceptance& acceptance);

}  // namespace wslam::v2::mapping::loop_closure
