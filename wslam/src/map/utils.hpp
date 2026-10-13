#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>

#include <Eigen/Core>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "data/provider_base.hpp"

// Helpers shared by the v1 and v2 mapping stages: IMU initialisation and
// preintegration, and the env-var overrides of the back-end knobs
namespace wslam::map::utils {

// The quietest window of an IMU buffer, see FindStationaryWindow
struct StationaryWindow {
    std::span<const data::IMUReading> samples;
    double score = std::numeric_limits<double>::infinity();
};

// Slides a window of window_size samples (half-window stride) over readings and
// returns the quietest one: population standard deviation of the acceleration
// magnitude plus the mean angular-velocity deviation from the window mean.
// Fails when even the best window scores max_score or more
[[nodiscard]] std::expected<StationaryWindow, std::string> FindStationaryWindow(
    std::span<const data::IMUReading> readings, size_t window_size,
    double max_score);

// Mean accelerometer and gyroscope readings, IMU frame
[[nodiscard]] std::pair<Eigen::Vector3d, Eigen::Vector3d> MeanAccelGyroImu(
    std::span<const data::IMUReading> readings);

// Gravity in the camera (= world) frame and the gyroscope bias (IMU frame)
// measured over a stationary window
[[nodiscard]] std::pair<Eigen::Vector3d, Eigen::Vector3d>
RecoverGravityAndGyroBias(const StationaryWindow& window,
                          const data::CamSensorParams& cam,
                          const data::IMUSensorParams& imu);

// Preintegration parameters from the IMU noise densities, the startup gravity
// estimate and the camera/IMU extrinsics
[[nodiscard]] std::shared_ptr<gtsam::PreintegrationCombinedParams>
CreateImuCalibration(const data::CamSensorParams& cam,
                     const data::IMUSensorParams& imu,
                     const Eigen::Vector3d& gravity,
                     double imu_integration_sigma);

// Integrates the samples in (prev_kf_ts, curr_kf_ts] into combined, plus a
// tail up to curr_kf_ts. Returns the number of integration steps
uint32_t IntegrateImuWindow(gtsam::PreintegratedCombinedMeasurements& combined,
                            std::span<const data::IMUReading> readings,
                            uint64_t prev_kf_ts, uint64_t curr_kf_ts);

// Env-var overrides for the mapping back-end's performance/accuracy levers, so
// a benchmark sweep can flip each knob without recompiling. Unset or empty
// variables, and values that don't parse, give the fallback
[[nodiscard]] double EnvDouble(const char* name, double fallback);
[[nodiscard]] int EnvInt(const char* name, int fallback);
[[nodiscard]] uint32_t EnvU32(const char* name, uint32_t fallback);
// WSLAM_ISAM_FACTORIZATION: CHOLESKY => true, QR => false
[[nodiscard]] bool EnvCholesky(bool fallback);

}  // namespace wslam::map::utils
