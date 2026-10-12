#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "match_scores.hpp"
#include "models.hpp"
#include "util/epipolar_geometry.hpp"

namespace wslam::v2 {

// Recovers the relative pose between the previous and the current frame from
// the RANSAC inliers and triangulates them.
//
// Picks the decomposition of RANSAC's essential matrix with the most inliers
// in front of both cameras, optionally refines it non-linearly (LightGlue
// score-weighted Sampson error, see util::RefineRelativePose), triangulates
// with linear DLT and keeps points that pass a depth and a reprojection
// filter. The translation is unit length, so are the landmark depths.
//
// Input  (from AnyBag): KeypointRansacResult under KeypointRansacName, camera
//                       parameters
// Output (to   AnyBag): KeypointTriangulationResult under
//                       KeypointTriangulationName
class TriangulatePass : public compute::Pass {
   public:
    struct Opts {
        // Reject landmarks whose mean reprojection error over both views
        // (LOD-0 pixels) exceeds this
        double max_reprojection_px = 4.0;

        // Keep landmarks in this depth range in both cameras, in units of
        // the unit translation
        double min_depth = 1e-3;
        double max_depth = 1e3;

        // Fewer inliers than this is no pose. The eight-point fit needs 8
        size_t min_inliers = 8;

        // Refine the decomposed pose before triangulating, with the
        // residuals weighted by `weighting`. The linear fit minimises an
        // algebraic error; on V101 refining roughly halves the rotation and
        // the p90 direction error (benchmarks/POSE_ESTIMATE_V2_NOTE.md)
        bool refine = true;
        ScoreWeighting weighting = ScoreWeighting::kScore;
        util::RefineOptions refine_options;

        uint32_t camera_index = 0;
    };

    TriangulatePass(AnyBag& storage, Opts opts)
        : storage_(storage), opts_(opts) {}

    // Forwarded to the Opts-taking constructor so the defaults live in Opts
    explicit TriangulatePass(AnyBag& storage);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    Opts opts_;

    // Set by initialize()
    std::optional<data::CamSensorParams> camera_;
};

// Body of TriangulatePass::execute
[[nodiscard]] KeypointTriangulationResult RunTriangulation(
    const KeypointRansacResult& ransac, const data::CamSensorParams& camera,
    const TriangulatePass::Opts& opts);

}  // namespace wslam::v2
