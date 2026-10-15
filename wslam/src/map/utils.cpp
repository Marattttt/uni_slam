#include "utils.hpp"

#include <gtsam/geometry/Pose3.h>
#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <ranges>

using namespace wslam;
using namespace wslam::map;

#define LOG_ID "[Mapping utils]"

double utils::StationaryScore(std::span<const data::IMUReading> window) {
    if (window.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    // Quietness of one window: population standard deviation of the
    // acceleration magnitude, plus the mean angular-velocity deviation from
    // the window's own mean. Subtracting the window mean makes the gyroscope
    // term immune to a constant bias, and using acceleration *magnitude*
    // makes the accelerometer term immune to a constant gravity vector, so a
    // parked vehicle scores near zero.
    const auto sample_count = static_cast<double>(window.size());

    Eigen::Vector3d mean_angular_velocity = Eigen::Vector3d::Zero();
    double acceleration_magnitude_sum = 0.0;
    double acceleration_magnitude_squared_sum = 0.0;
    for (const data::IMUReading& reading : window) {
        mean_angular_velocity
            += Eigen::Vector3d(static_cast<double>(reading.wx()),
                               static_cast<double>(reading.wy()),
                               static_cast<double>(reading.wz()));
        const double acceleration_magnitude
            = Eigen::Vector3d(static_cast<double>(reading.ax()),
                              static_cast<double>(reading.ay()),
                              static_cast<double>(reading.az()))
                  .norm();
        acceleration_magnitude_sum += acceleration_magnitude;
        acceleration_magnitude_squared_sum
            += acceleration_magnitude * acceleration_magnitude;
    }
    mean_angular_velocity /= sample_count;

    double angular_velocity_deviation_sum = 0.0;
    for (const data::IMUReading& reading : window) {
        const Eigen::Vector3d angular_velocity(
            static_cast<double>(reading.wx()),
            static_cast<double>(reading.wy()),
            static_cast<double>(reading.wz()));
        angular_velocity_deviation_sum
            += (angular_velocity - mean_angular_velocity).norm();
    }

    const double mean_acceleration_magnitude
        = acceleration_magnitude_sum / sample_count;
    const double acceleration_magnitude_variance
        = std::max(0.0, acceleration_magnitude_squared_sum / sample_count
                            - mean_acceleration_magnitude
                                  * mean_acceleration_magnitude);

    return std::sqrt(acceleration_magnitude_variance)
           + angular_velocity_deviation_sum / sample_count;
}

std::expected<utils::StationaryWindow, std::string> utils::FindStationaryWindow(
    std::span<const data::IMUReading> readings, size_t window_size,
    double max_score) {
    if (window_size == 0) {
        return std::unexpected("window size 0");
    }
    if (readings.size() == 0) {
        return std::unexpected("buffer size 0");
    }

    const auto step = std::max(1UZ, window_size / 2);
    const auto windows
        = readings | std::views::slide(window_size) | std::views::stride(step);

    StationaryWindow best{.samples = readings};
    for (const auto& chunk : windows) {
        // chunk leaves a short final window when the buffer isn't an exact
        // multiple of window_size. A one- or two-sample window has almost no
        // spread and would win with a meaningless score, so skip it -- unless
        // the buffer is smaller than a single window, in which case that lone
        // short chunk *is* the whole buffer and we score it, recovering the
        // old "average everything" fallback.
        const bool is_full_window = chunk.size() == window_size;
        if (!is_full_window && readings.size() >= window_size) {
            continue;
        }
        const std::span<const data::IMUReading> window(chunk);
        const double score = StationaryScore(window);
        if (score < best.score) {
            best = {.samples = window, .score = score};
        }
    }

    if (best.score >= max_score) {
        return std::unexpected(std::format(
            "no window quiet enough to be stationary (best score {:.4f} >= "
            "{:.4f}; best window {} samples of {} total)",
            best.score, max_score, best.samples.size(), readings.size()));
    }
    return best;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> utils::MeanAccelGyroImu(
    std::span<const data::IMUReading> readings) {
    Eigen::Vector3f accel_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_sum = Eigen::Vector3f::Zero();

    for (const auto& r : readings) {
        accel_sum += Eigen::Vector3f{r.ax(), r.ay(), r.az()};
        gyro_sum += Eigen::Vector3f{r.wx(), r.wy(), r.wz()};
    }

    Eigen::Vector3d accel
        = accel_sum.cast<double>() / static_cast<double>(readings.size());
    Eigen::Vector3d gyro
        = gyro_sum.cast<double>() / static_cast<double>(readings.size());

    return {accel, gyro};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> utils::RecoverGravityAndGyroBias(
    const StationaryWindow& window, const data::CamSensorParams& cam,
    const data::IMUSensorParams& imu) {
    // The mean accel is in the IMU sensor frame; gravity must end up in the
    // camera (= world) frame. Both extrinsics are sensor->body (T_BS), so the
    // IMU->camera rotation is R(T_body_cam^-1 * T_body_imu) =
    // R_body_cam^T * R_body_imu. Composing imu.T_BS keeps this correct for any
    // provider.
    const Eigen::Matrix3d r_body_cam = cam.T_BS.block<3, 3>(0, 0);
    const Eigen::Matrix3d r_body_imu = imu.T_BS.block<3, 3>(0, 0);
    const Eigen::Matrix3d r_cam_imu = r_body_cam.transpose() * r_body_imu;

    const auto [accel_mean, gyro_mean] = MeanAccelGyroImu(window.samples);

    // Specific force on a stationary IMU is the negative of gravity; rotate the
    // negated mean accel from the IMU frame into the camera/world frame.
    const auto gravity = r_cam_imu * -accel_mean;

    // The mean gyro reading of a stationary IMU is its bias (kept in the IMU
    // frame, which is where the CombinedImuFactor expects it).
    const auto gyro_bias = gyro_mean;

    return {gravity, gyro_bias};
}

std::optional<utils::StationaryWindow> utils::FindLastStationaryWindow(
    std::span<const data::IMUReading> readings, size_t window_size,
    double max_score) {
    if (window_size == 0 || readings.size() < window_size) {
        return std::nullopt;
    }

    const auto step = std::max(1UZ, window_size / 2);
    for (size_t end = readings.size();; end -= step) {
        const auto window = readings.subspan(end - window_size, window_size);
        const double score = StationaryScore(window);
        if (score < max_score) {
            return StationaryWindow{.samples = window, .score = score};
        }
        if (end < window_size + step) {
            return std::nullopt;
        }
    }
}

utils::StartupImuState utils::RecoverStartupImuState(
    const StationaryWindow& window, const data::CamSensorParams& cam,
    const data::IMUSensorParams& imu, double gravity_magnitude) {
    const Eigen::Matrix3d r_body_cam = cam.T_BS.block<3, 3>(0, 0);
    const Eigen::Matrix3d r_body_imu = imu.T_BS.block<3, 3>(0, 0);
    const Eigen::Matrix3d r_cam_imu = r_body_cam.transpose() * r_body_imu;

    const auto [accel_mean, gyro_mean] = MeanAccelGyroImu(window.samples);

    // At rest the specific force is -gravity + accel bias (IMU frame). Its
    // direction is taken as gravity's; what exceeds the nominal magnitude
    // along it is bias
    const Eigen::Vector3d gravity_imu
        = -accel_mean.normalized() * gravity_magnitude;
    const Eigen::Vector3d accel_bias = accel_mean + gravity_imu;

    return StartupImuState{
        .gravity_cam = r_cam_imu * gravity_imu,
        .bias = gtsam::imuBias::ConstantBias{accel_bias, gyro_mean},
    };
}

utils::PropagatedRest utils::PropagateFromRest(
    const Eigen::Vector3d& gravity_cam,
    std::span<const data::IMUReading> readings, uint64_t from_ts,
    uint64_t to_ts, const gtsam::imuBias::ConstantBias& bias,
    const data::CamSensorParams& cam, const data::IMUSensorParams& imu) {
    if (to_ts <= from_ts) {
        return {.gravity_cam = gravity_cam};
    }

    // Navigation frame: the camera at from_ts, where gravity_cam holds
    gtsam::PreintegratedCombinedMeasurements combined{
        CreateImuCalibration(cam, imu, gravity_cam, 1e-4), bias};

    const auto first = std::ranges::find_if(
        readings, [from_ts](const auto& r) { return r.timestamp > from_ts; });
    const auto last = std::find_if(first, readings.end(), [to_ts](auto& r) {
        return r.timestamp > to_ts;
    });
    const std::span<const data::IMUReading> window{first, last};
    if (window.empty()) {
        return {.gravity_cam = gravity_cam};
    }
    IntegrateImuWindow(combined, window, from_ts, to_ts);

    const gtsam::NavState state
        = combined.predict(gtsam::NavState{gtsam::Rot3{}, gtsam::Point3::Zero(),
                                           gtsam::Velocity3::Zero()},
                           bias);

    // The world does not move: both seen from the camera at to_ts
    const Eigen::Matrix3d r_from_to = state.pose().rotation().matrix();
    return {
        .gravity_cam = r_from_to.transpose() * gravity_cam,
        .velocity_cam = r_from_to.transpose() * state.v(),
    };
}

std::shared_ptr<gtsam::PreintegrationCombinedParams>
utils::CreateImuCalibration(const data::CamSensorParams& cam,
                            const data::IMUSensorParams& imu,
                            const Eigen::Vector3d& gravity,
                            double imu_integration_sigma) {
    const double gyro_var
        = imu.gyroscope_noise_density * imu.gyroscope_noise_density;
    const double accel_var
        = imu.accelerometer_noise_density * imu.accelerometer_noise_density;
    const double gyro_bias_var
        = imu.gyroscope_random_walk * imu.gyroscope_random_walk;
    const double accel_bias_var
        = imu.accelerometer_random_walk * imu.accelerometer_random_walk;

    auto calib = gtsam::PreintegrationCombinedParams::MakeSharedU();
    calib->n_gravity = gravity;
    calib->setGyroscopeCovariance(gtsam::Matrix3::Identity() * gyro_var);
    calib->setAccelerometerCovariance(gtsam::Matrix3::Identity() * accel_var);
    calib->setIntegrationCovariance(
        gtsam::Matrix3::Identity()
        * (imu_integration_sigma * imu_integration_sigma));
    calib->setBiasOmegaCovariance(gtsam::Matrix3::Identity() * gyro_bias_var);
    calib->setBiasAccCovariance(gtsam::Matrix3::Identity() * accel_bias_var);
    calib->setBiasAccOmegaInit(gtsam::Matrix6::Identity() * 1e-5);

    // body_P_sensor = pose of the IMU in the navigation body frame, which this
    // pipeline takes to be the camera frame (the pose variables are camera
    // poses). Both extrinsics use the sensor->body T_BS convention, so:
    //   T_cam_imu = T_body_cam^-1 * T_body_imu
    // Composing imu.T_BS keeps this correct for any provider, including one
    // whose IMU frame differs from its body frame.
    //
    // Remark: a provider whose body frame coincides with its IMU frame (as in
    // EuRoC) must set imu.T_BS to identity, which collapses this to
    // T_body_cam^-1.
    const gtsam::Pose3 T_body_cam(cam.T_BS);
    const gtsam::Pose3 T_body_imu(imu.T_BS);
    calib->setBodyPSensor(T_body_cam.inverse() * T_body_imu);
    return calib;
}

uint32_t utils::IntegrateImuWindow(
    gtsam::PreintegratedCombinedMeasurements& combined,
    std::span<const data::IMUReading> readings, uint64_t prev_kf_ts,
    uint64_t curr_kf_ts) {
    uint64_t last_ts = prev_kf_ts;
    uint32_t integrated = 0;

    const auto integrate
        = [&combined, &integrated](const data::IMUReading& r, double dt_s) {
              const gtsam::Vector3 accel{r.ax(), r.ay(), r.az()};
              const gtsam::Vector3 gyro{r.wx(), r.wy(), r.wz()};
              combined.integrateMeasurement(accel, gyro, dt_s);
              integrated++;
          };

    const auto is_good_ts
        = [prev_kf_ts](auto&& r) { return r.timestamp > prev_kf_ts; };

    size_t skipped_frames = 0;
    for (const auto& r : readings | std::views::filter(is_good_ts)) {
        if (r.timestamp > last_ts
            && static_cast<double>(r.timestamp - last_ts) > 1e9) {
            skipped_frames++;
        }

        const auto dt_s = static_cast<double>(r.timestamp - last_ts) * 1e-9;
        last_ts = r.timestamp;
        integrate(r, dt_s);
    }

    if (skipped_frames > 0) {
        spdlog::warn(LOG_ID " Skipped {} frames when integrating IMU",
                     skipped_frames);
    }

    // Tail up to the keyframe timestamp. Guard against an empty window —
    // readings.back() on an empty span is UB.
    if (!readings.empty() && curr_kf_ts > last_ts) {
        const double dt = static_cast<double>(curr_kf_ts - last_ts) * 1e-9;
        integrate(readings.back(), dt);
    }

    return integrated;
}

namespace {
const char* EnvOrNull(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : nullptr;
}
}  // namespace

double utils::EnvDouble(const char* name, double fallback) {
    if (const char* value = EnvOrNull(name)) {
        try {
            return std::stod(value);
        } catch (...) {
            spdlog::warn(LOG_ID " ignoring non-numeric {}='{}'", name, value);
        }
    }
    return fallback;
}

int utils::EnvInt(const char* name, int fallback) {
    if (const char* value = EnvOrNull(name)) {
        try {
            return std::stoi(value);
        } catch (...) {
            spdlog::warn(LOG_ID " ignoring non-integer {}='{}'", name, value);
        }
    }
    return fallback;
}

uint32_t utils::EnvU32(const char* name, uint32_t fallback) {
    const int value = EnvInt(name, static_cast<int>(fallback));
    return value < 0 ? fallback : static_cast<uint32_t>(value);
}

bool utils::EnvCholesky(bool fallback) {
    if (const char* value = EnvOrNull("WSLAM_ISAM_FACTORIZATION")) {
        const std::string s(value);
        if (s == "CHOLESKY" || s == "cholesky") {
            return true;
        }
        if (s == "QR" || s == "qr") {
            return false;
        }
        spdlog::warn(LOG_ID " ignoring unknown WSLAM_ISAM_FACTORIZATION='{}'",
                     value);
    }
    return fallback;
}
