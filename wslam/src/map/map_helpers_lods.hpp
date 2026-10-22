#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>

#include "common.hpp"
#include "data/provider_base.hpp"
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
inline Eigen::Vector2d UndistortNormalised(const Eigen::Vector2d& dist,
                                           const Eigen::Vector4d& d,
                                           uint32_t max_iters
                                           = kDefaultUndistortMaxIters,
                                           double tol
                                           = kDefaultUndistortTolerance) {
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

// Forward equidistant (Kannala-Brandt / OpenCV fisheye) distortion, as Kalibr
// defines it: the incidence angle theta = atan(r) of the ideal point maps to
//   theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
// and the distorted point keeps the ideal one's direction at radius theta_d
inline Eigen::Vector2d DistortEquidistant(const Eigen::Vector2d& xy,
                                          const Eigen::Vector4d& d) {
    const double r = xy.norm();
    if (r < std::numeric_limits<double>::epsilon()) {
        return xy;
    }
    const double theta = std::atan(r);
    const double theta2 = theta * theta;
    const double theta_d
        = theta
          * (1.0
             + theta2
                   * (d[0]
                      + theta2 * (d[1] + theta2 * (d[2] + theta2 * d[3]))));
    return xy * (theta_d / r);
}

// Inverse of DistortEquidistant: Newton on theta_d(theta), from theta =
// theta_d. NaN when the solution reaches 90 degrees, a ray the normalised
// image plane can't hold
inline Eigen::Vector2d UndistortEquidistant(const Eigen::Vector2d& dist,
                                            const Eigen::Vector4d& d,
                                            uint32_t max_iters
                                            = kDefaultUndistortMaxIters,
                                            double tol
                                            = kDefaultUndistortTolerance) {
    const double theta_d = dist.norm();
    if (theta_d < std::numeric_limits<double>::epsilon()) {
        return dist;
    }
    double theta = theta_d;
    for (uint32_t i = 0; i < max_iters; ++i) {
        const double theta2 = theta * theta;
        const double value
            = theta
                  * (1.0
                     + theta2
                           * (d[0]
                              + theta2
                                    * (d[1] + theta2 * (d[2] + theta2 * d[3]))))
              - theta_d;
        const double slope
            = 1.0
              + theta2
                    * (3.0 * d[0]
                       + theta2
                             * (5.0 * d[1]
                                + theta2 * (7.0 * d[2] + theta2 * 9.0 * d[3])));
        const double step = value / slope;
        theta -= step;
        if (std::abs(step) < tol) {
            break;
        }
    }
    if (!(theta >= 0.0 && theta < std::numbers::pi / 2.0)) {
        return Eigen::Vector2d::Constant(
            std::numeric_limits<double>::quiet_NaN());
    }
    return dist * (std::tan(theta) / theta_d);
}

// The camera's distortion model applied to an ideal normalised point
inline Eigen::Vector2d DistortNormalised(const Eigen::Vector2d& xy,
                                         const data::CamSensorParams& camera) {
    switch (camera.distortion_model) {
        case data::DistortionModel::Equidistant:
            return DistortEquidistant(xy, camera.distortion_coefficients);
        case data::DistortionModel::RadialTangential:
            break;
    }
    return DistortNormalised(xy, camera.distortion_coefficients);
}

// Inverse of the camera's distortion model
inline Eigen::Vector2d UndistortNormalised(const Eigen::Vector2d& dist,
                                           const data::CamSensorParams& camera,
                                           uint32_t max_iters
                                           = kDefaultUndistortMaxIters,
                                           double tol
                                           = kDefaultUndistortTolerance) {
    switch (camera.distortion_model) {
        case data::DistortionModel::Equidistant:
            return UndistortEquidistant(dist, camera.distortion_coefficients,
                                        max_iters, tol);
        case data::DistortionModel::RadialTangential:
            break;
    }
    return UndistortNormalised(dist, camera.distortion_coefficients, max_iters,
                               tol);
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

// Feature coordinates are in the pixels of the feature's own pyramid level;
// scales them up to LOD-0 pixels
constexpr Eigen::Vector2d ToLod0Pixel(const Feature& f) {
    const double scale = LodScale(f.lod);
    return {static_cast<double>(f.x) * scale, static_cast<double>(f.y) * scale};
}

// Maps a LOD-0 pixel through the inverse intrinsic + inverse distortion
// pipeline, yielding a 2D normalised-image-plane (calibrated) coordinate.
inline Eigen::Vector2d PixelToNormalised(const Eigen::Vector2d& pixel,
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

// PixelToNormalised under the camera's own distortion model
inline Eigen::Vector2d PixelToNormalised(const Eigen::Vector2d& pixel,
                                         const data::CamSensorParams& camera,
                                         uint32_t max_iters, double tol) {
    const Eigen::Vector4d& k = camera.intrinsics;
    const Eigen::Vector2d dist((pixel.x() - k[2]) / k[0],
                               (pixel.y() - k[3]) / k[1]);
    return UndistortNormalised(dist, camera, max_iters, tol);
}
}  // namespace wslam::map::util
