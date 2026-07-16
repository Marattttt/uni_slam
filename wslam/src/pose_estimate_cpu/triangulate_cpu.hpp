#pragma once

#include <cstdint>

#include "common.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"

namespace wslam {

// CPU triangulation pass.
//
// Inputs  (from AnyBag):
//   - RansacResult under ResourceIdentifier::RansacResultName
//   - data::CamSensorParams under ResourceIdentifier::GetCameraIntrinsicsName(0)
//   - Optional std::vector<data::IMUReading> under
//     ResourceIdentifier::GetImuVecName() — used only for a sanity-check log
//     line comparing the gyro-integrated rotation magnitude to the geometric
//     rotation we recover.
//
// Output (to AnyBag):
//   - TriangulationResult under ResourceIdentifier::TriangulationResultName
//
// The pass undistorts the LOD-0 pixel coordinates of every RANSAC inlier,
// re-fits a clean essential matrix on the resulting calibrated rays, picks the
// relative pose hypothesis with the most points in front of both cameras, then
// runs a linear DLT triangulation followed by a reprojection-error filter.
class TriangulateCPU : public compute::Pass {
   public:
    struct Opts {
        // Reject any landmark whose mean reprojection error (LOD-0 pixels)
        // exceeds this. 4 px is comfortable given the LOD-0 image resolution
        // and the upstream Sampson-distance threshold of 3.84 px^2.
        double max_reprojection_px = 4.0;

        // Drop landmarks too close to or behind either camera. The chosen
        // scale is unit translation, so depths are dimensionless multiples.
        double min_depth = 1e-3;
        double max_depth = 1e3;

        // Below this many inliers the pass writes an empty result and skips
        // pose recovery. Eight is the minimum for the eight-point algorithm.
        size_t min_inliers = 8;

        // Maximum Newton iterations for inverse radial-tangential distortion.
        // Convergence is typically reached in 4–6 iterations.
        uint32_t undistort_iterations = 20;

        // Convergence tolerance on the normalised-coordinate residual.
        double undistort_tolerance = 1e-9;

        // Index of the camera (in SensorParams::cams) we triangulate against.
        uint32_t camera_index = 0;
    };

    TriangulateCPU(AnyBag& storage, Opts opts);

    // Forwarded to the Opts-taking constructor so default values aren't
    // duplicated.
    TriangulateCPU(AnyBag& storage);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    Opts opts_;

    // Cached on first execute() so we don't pay the AnyBag lookup every frame.
    std::optional<data::CamSensorParams> intrinsics_;

    [[nodiscard]] std::optional<std::string> ensureIntrinsics();
};
}  // namespace wslam
