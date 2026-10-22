#pragma once

#include <Eigen/Core>
#include <expected>
#include <format>
#include <limits>
#include <string>

#include "anybag.hpp"
#include "common.hpp"
#include "data/provider_base.hpp"
#include "map/map_helpers_lods.hpp"

// Pinhole camera helpers (radial-tangential or equidistant distortion) for the
// v2 pose estimate passes
namespace wslam::v2::util {

// Camera parameters the sensor intrinsics were published under at pipeline
// creation
[[nodiscard]] inline std::expected<data::CamSensorParams, std::string>
ReadCameraParams(const AnyBag& storage, uint32_t camera_index) {
    const auto key = ResourceIdentifier::GetCameraIntrinsicsName(camera_index);
    auto params = storage.get<data::CamSensorParams>(key);
    if (!params) {
        return std::unexpected(
            std::format("no camera parameters under '{}'", key));
    }
    return std::move(params).value();
}

// Pixel of a point on the normalised image plane (applies K only). For an
// undistorted ray this is the pixel an ideal pinhole camera would observe
[[nodiscard]] inline Eigen::Vector2d NormalisedToPixel(
    const Eigen::Vector2d& normalised, const data::CamSensorParams& camera) {
    const Eigen::Vector4d& k = camera.intrinsics;  // fu, fv, cu, cv
    return {k[0] * normalised.x() + k[2], k[1] * normalised.y() + k[3]};
}

// Observed (distorted) pixel of a point in camera coordinates. NaN behind
// the camera
[[nodiscard]] inline Eigen::Vector2d ProjectToPixel(
    const Eigen::Vector3d& point, const data::CamSensorParams& camera) {
    if (point.z() <= 0.0) {
        return Eigen::Vector2d::Constant(
            std::numeric_limits<double>::quiet_NaN());
    }
    const Eigen::Vector2d distorted
        = map::util::DistortNormalised(point.head<2>() / point.z(), camera);
    return NormalisedToPixel(distorted, camera);
}

}  // namespace wslam::v2::util
