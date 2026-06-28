#pragma once

#include <Eigen/Dense>
#include <ranges>

#include "common.hpp"
#include "models.hpp"

namespace wslam::map::util {
// Forward radial-tangential distortion as defined by OpenCV / EuRoC:
//   x'' = x' (1 + k1 r^2 + k2 r^4) + 2 p1 x' y' + p2 (r^2 + 2 x'^2)
//   y'' = y' (1 + k1 r^2 + k2 r^4) + p1 (r^2 + 2 y'^2) + 2 p2 x' y'
//
// (x', y') are normalised (calibrated) ideal coords; (x'', y'') are the
// normalised coords that would actually hit the imager.
constexpr Eigen::Vector2d DistortNormalised(const Eigen::Vector2d& xy,
                                            const Eigen::Vector4d& d) {
    const double x = xy.x();
    const double y = xy.y();
    const double r2 = x * x + y * y;
    const double k1 = d[0];
    const double k2 = d[1];
    const double p1 = d[2];
    const double p2 = d[3];
    const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    return {x * radial + dx, y * radial + dy};
}

// Returns the rotation angle (radians) of a 3x3 rotation, clamped for
// numerical safety. Mirrors the helper in triangulate_cpu.cpp.
constexpr double ComputeRotationAngle(const Eigen::Matrix3d& r) {
    const double trace = std::clamp(r.trace(), -1.0, 3.0);
    return std::acos(std::clamp(0.5 * (trace - 1.0), -1.0, 1.0));
};

constexpr uint32_t kDefaultUndistortMaxIters = 8;
constexpr double kDefaultUndistortTolerance = 1e-9;

// Inverse model: solves DistortNormalised(undist) = dist via Newton-style
// fixed-point iteration. The standard OpenCV trick: start with the distorted
// point as the initial guess (small distortion ⇒ near-identity), and update
// by subtracting the residual scaled by the inverse radial term.
Eigen::Vector2d UndistortNormalised(const Eigen::Vector2d& dist,
                                    const Eigen::Vector4d& d,
                                    uint32_t max_iters
                                    = kDefaultUndistortMaxIters,
                                    double tol = kDefaultUndistortTolerance) {
    Eigen::Vector2d xy = dist;
    for (uint32_t i = 0; i < max_iters; ++i) {
        const double x = xy.x();
        const double y = xy.y();
        const double r2 = x * x + y * y;
        const double k1 = d[0];
        const double k2 = d[1];
        const double p1 = d[2];
        const double p2 = d[3];
        const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        const double dx_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        const double dy_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        // The standard inversion update:
        //   xy_next = (dist - tangential) / radial
        // — derived by assuming the radial term is approximately constant
        // across the iteration.
        const Eigen::Vector2d next((dist.x() - dx_tan) / radial,
                                   (dist.y() - dy_tan) / radial);
        const double step = (next - xy).norm();
        xy = next;
        if (step < tol) {
            break;
        }
    }
    return xy;
}

constexpr double LodScale(std::unsigned_integral auto lod) {
    static constinit auto lod_factors = std::invoke([]() {
        const auto factors = std::views::iota(0)
                             | std::views::take(GPUConst::levels_of_detail)
                             | std::views::transform([](auto&& lod) {
                                   double factor = 1;
                                   for (int i = 0; i < lod; i++) {
                                       factor *= GPUConst::lod_scale_factor;
                                   }
                                   return factor;
                               });

        std::array<double, GPUConst::levels_of_detail> arr;
        std::ranges::copy(factors, arr.begin());
        return arr;
    });

    return lod_factors.at(static_cast<size_t>(lod));
}

constexpr Eigen::Vector2d ToLodPixel(const Feature& f,
                                     std::unsigned_integral auto lod) {
    const double scale = LodScale(lod);
    return {static_cast<double>(f.x) * scale, static_cast<double>(f.y) * scale};
}

constexpr Eigen::Vector2d ToLod0Pixel(const Feature& f) {
    return ToLodPixel(f, 0U);
}

// Maps a LOD-0 pixel through the inverse intrinsic + inverse distortion
// pipeline, yielding a 2D normalised-image-plane (calibrated) coordinate.
Eigen::Vector2d PixelToNormalised(const Eigen::Vector2d& pixel,
                                  const Eigen::Vector4d& intrinsics,
                                  const Eigen::Vector4d& distortion,
                                  uint32_t max_iters, double tol) {
    const double fu = intrinsics[0];
    const double fv = intrinsics[1];
    const double cu = intrinsics[2];
    const double cv = intrinsics[3];
    const Eigen::Vector2d dist((pixel.x() - cu) / fu, (pixel.y() - cv) / fv);
    return UndistortNormalised(dist, distortion, max_iters, tol);
}
}  // namespace wslam::map::util
