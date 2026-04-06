#include "euroc_provider.hpp"

#include <csv.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>

#include "provider_base.hpp"
#include "yaml-cpp/exceptions.h"

using namespace wslam::data;
namespace fs = std::filesystem;

#define LOG_ID "[EuRoC provider]"

namespace {
Eigen::Matrix4d parse4x4(const YAML::Node& node) noexcept(false) {
    auto vals = node.as<std::vector<double>>();

    Eigen::Matrix4d mat{};

    for (size_t i = 0; i < vals.size(); i++) {
        const auto idx = static_cast<Eigen::Index>(i);
        mat(idx / 4, idx % 4) = vals[i];
    }

    return mat;
}

Eigen::Vector4d parse1x4(const YAML::Node& node) noexcept(false) {
    auto vals = node.as<std::array<double, 4>>();
    Eigen::Vector4d vec{};

    for (size_t i = 0; i < vals.size(); i++) {
        vec[static_cast<Eigen::Index>(i)] = vals[i];
    }

    return vec;
}

std::expected<CamSensorParams, std::string> getCamSensorParamsSingle(
    const std::filesystem::path& file_path) {
    // clang-format off
/*
# General sensor definitions.
sensor_type: camera
comment: VI-Sensor cam0 (MT9M034)

# Sensor extrinsics wrt. the body-frame.
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]

# Camera specific definitions.
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
*/
    // clang-format on

    YAML::Node sensor = YAML::LoadFile(file_path);
    CamSensorParams params;
    try {
        YAML::Node tbs = sensor["T_BS"];
        params.T_BS = parse4x4(tbs["data"]);
        params.rate_hz = sensor["rate_hz"].as<uint32_t>();
        params.resolution = sensor["resolution"].as<std::array<int, 2>>();
        params.camera_model = sensor["camera_model"].as<std::string>();
        params.intrinsics = parse1x4(sensor["intrinsics"]);
        params.distortion_model = sensor["distortion_model"].as<std::string>();
        params.distortion_coefficients
            = parse1x4(sensor["distortion_coefficients"]);

    } catch (YAML::InvalidNode& err) {
        return std::unexpected(
            std::format("parsing yaml (invalid node): {}", err.what()));
    } catch (YAML::InvalidScalar& err) {
        return std::unexpected(
            std::format("parsing yaml (invalid scalar): {}", err.what()));
    } catch (std::exception& err) {
        return std::unexpected(std::format("unexpected error: {}", err.what()));
    }

    return params;
}
std::expected<IMUSensorParams, std::string> getIMUSensorParamsSingle(
    const fs::path& path) {
    // clang-format off
/*
#Default imu sensor yaml file
sensor_type: imu
comment: VI-Sensor IMU (ADIS16448)

# Sensor extrinsics wrt. the body-frame.
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 200 

# inertial sensor noise model parameters (static)
gyroscope_noise_density: 1.6968e-04     # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" ) 
gyroscope_random_walk: 1.9393e-05       # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
accelerometer_noise_density: 2.0000e-3  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
accelerometer_random_walk: 3.0000e-3    # [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )
*/
    // clang-format on

    IMUSensorParams params;

    try {
        YAML::Node sensor = YAML::LoadFile(path);

        YAML::Node tbs = sensor["T_BS"];
        params.T_BS = parse4x4(tbs["data"]);

        params.rate_hz = sensor["rate_hz"].as<uint32_t>();
        params.gyroscope_noise_density
            = sensor["gyroscope_noise_density"].as<double>();
        params.gyroscope_random_walk
            = sensor["gyroscope_random_walk"].as<double>();
        params.accelerometer_noise_density
            = sensor["accelerometer_noise_density"].as<double>();
        params.accelerometer_random_walk
            = sensor["accelerometer_random_walk"].as<double>();
    } catch (YAML::InvalidNode& err) {
        return std::unexpected(
            std::format("parsing yaml (invalid node): {}", err.what()));
    } catch (YAML::InvalidScalar& err) {
        return std::unexpected(
            std::format("parsing yaml (invalid scalar): {}", err.what()));
    } catch (std::exception& err) {
        return std::unexpected(std::format("unexpected error: {}", err.what()));
    }

    return params;
}
};  // namespace

EurocProviderOpts wslam::data::CreateEurocProviderOpts() {
    EurocProviderOpts opts;
    const char* dir = std::getenv(EUROC_DIR_ENV);
    if (dir == nullptr) {
        opts.path = ".";
        spdlog::warn("Env variable " EUROC_DIR_ENV
                     " empty or unset. Using '.' as the default path");
    } else {
        opts.path = dir;
    }

    return opts;
}

std::optional<std::string> EurocProvider::updateIMUSensorParams() {
    const fs::path path = kImuDir / "sensor.yaml";

    auto val = getIMUSensorParamsSingle(path);
    if (val) {
        imu_params_ = std::make_unique<IMUSensorParams>(val.value());
        return std::nullopt;
    }

    return val.error();
}

std::optional<std::string> EurocProvider::updateCamSensorParams() {
    const fs::path cam0 = kCam0Dir / "sensor.yaml";
    const fs::path cam1 = kCam1Dir / "sensor.yaml";

    std::vector<CamSensorParams> params{};

    if (auto par0 = getCamSensorParamsSingle(cam0)) {
        params.emplace_back(std::move(par0.value()));
    } else {
        return std::format("cam0: {}", par0.error());
    }

    if (auto par1 = getCamSensorParamsSingle(cam1)) {
        params.emplace_back(std::move(par1.value()));
    } else {
        return std::format("cam1: {}", par1.error());
    }

    cam_params_ = std::make_unique<std::array<CamSensorParams, 2>>();
    (*cam_params_)[0] = params[0];
    (*cam_params_)[1] = params[1];

    return std::nullopt;
}

std::expected<SensorParams, std::string> EurocProvider::getSensorParams() {
    if (imu_params_ && cam_params_) {
        return SensorParams{.imu = *imu_params_,
                            .cams = {cam_params_->at(0), cam_params_->at(1)}};
    }
    SensorParams params;

    if (!imu_params_) {
        auto err = updateIMUSensorParams();
        if (err) {
            return std::unexpected(std::format("imu: {}", err.value()));
        }
    }

    if (!cam_params_) {
        auto err = updateCamSensorParams();
        if (err) {
            return std::unexpected(std::format("cam: {}", err.value()));
        }
    }

    return SensorParams{.imu = *imu_params_,
                        .cams = {cam_params_->at(0), cam_params_->at(1)}};
}

namespace {
std::expected<std::vector<IMUReading>, std::string> loadIMUReadings(
    const fs::path& path) {
    const int csvColumns = 7;

    uint64_t timestamp;
    Eigen::Vector<float, csvColumns - 1> values;

    std::vector<IMUReading> readings;

    try {
        spdlog::info("[EUROC privder] begin reading csv from {}",
                     path.string());

        io::CSVReader<csvColumns, io::trim_chars<' '>, io::no_quote_escape<','>,
                      io::throw_on_overflow, io::single_line_comment<'#'>>
            reader(path);

        // NOLINTBEGIN(readability-magic-numbers)
        while (reader.read_row(timestamp, values[0], values[1], values[2],
                               values[3], values[4], values[5])) {
            // NOLINTEND(readability-magic-numbers)
            readings.emplace_back(timestamp, std::move(values));
        }
    } catch (std::exception& err) {
        return std::unexpected(std::format("reading csv: {}", err.what()));
    }

    return readings;
}
};  // namespace

namespace {
std::expected<std::vector<EurocProvider::FrameInfo>, std::string> getFramePaths(
    const fs::path& dir) {
    std::vector<EurocProvider::FrameInfo> results;

    uint64_t ts_ns;
    std::string filename;

    const int csvColumns = 2;

    io::CSVReader<csvColumns, io::trim_chars<' '>, io::no_quote_escape<','>,
                  io::throw_on_overflow, io::single_line_comment<'#'>>
        csv(dir / "data.csv");

    while (csv.read_row(ts_ns, filename)) {
        const fs::path filepath = dir / "data" / filename;
        results.emplace_back(ts_ns, filepath);
    }

    std::ranges::sort(results, std::less{},
                      &EurocProvider::FrameInfo::timestamp);

    spdlog::info("[Euroc provider] found {} frames in directory {}",
                 results.size(), dir.string());

    return results;
}
}  // namespace

std::optional<std::string> EurocProvider::collectFramePaths() {
    auto cam0 = getFramePaths(kCam0Dir);
    if (!cam0) {
        return std::format("cam0: {}", cam0.error());
    }

    auto cam1 = getFramePaths(kCam1Dir);
    if (!cam1) {
        return std::format("cam1: {}", cam1.error());
    }

    cam0_paths_ = std::move(cam0.value());
    cam1_paths_ = std::move(cam1.value());

    return std::nullopt;
}

std::optional<std::string> EurocProvider::initialize() {
    spdlog::info("[Euroc provider] Initializing");

    try {
        const std::string kImuFilename = "data.csv";
        if (imu_readings_.empty()) {
            auto readings = loadIMUReadings(kImuDir / kImuFilename);
            if (!readings) {
                return std::format("getting imu readings: {}",
                                   readings.error());
            }
            imu_readings_ = std::move(readings.value());
        }
        if (cam0_paths_.empty() || cam1_paths_.empty()) {
            if (auto err = collectFramePaths()) {
                return std::format("collecting frames: {}", err.value());
            }
        }
    } catch (const std::exception& e) {
        return std::format("exception: {}", e.what());
    }

    return std::nullopt;
}

std::generator<std::expected<Reading<2>, std::string>>
EurocProvider::getReadings() {
    if (auto err = initialize()) {
        co_yield std::unexpected(
            std::format("initializing: {}", std::move(err.value())));
        co_return;
    }

    if (last_timestamp_ == 0) {
        const auto first_timestamp = std::min({imu_readings_.front().timestamp,
                                               cam0_paths_.front().timestamp,
                                               cam1_paths_.front().timestamp});
        last_timestamp_ = first_timestamp;
    }

    auto curr_imu = imu_readings_.cbegin();
    auto curr_cam0 = cam0_paths_.cbegin();
    auto curr_cam1 = cam1_paths_.cbegin();

    constexpr auto end = std::numeric_limits<size_t>::max();

    while (true) {
        const size_t imu_ts
            = curr_imu == imu_readings_.end() ? end : curr_imu->timestamp;
        const size_t cam0_ts
            = curr_cam0 == cam0_paths_.end() ? end : curr_cam0->timestamp;
        const size_t cam1_ts
            = curr_cam1 == cam1_paths_.end() ? end : curr_cam1->timestamp;

        const auto min = std::min({imu_ts, cam0_ts, cam1_ts});

        if (min == end) {
            co_return;
        }

        ReadingType reading;

        if (imu_ts == min) {
            reading->imu = *curr_imu;
            curr_imu++;
        }

        if (cam0_ts == min) {
            auto frame = getLocalFrame(curr_cam0->path, curr_cam0->timestamp);
            if (!frame) {
                spdlog::error(
                    LOG_ID " could not get frame for cam0 at ts:{}, path:{}",
                    curr_cam0->timestamp, std::string(curr_cam0->path));

                co_yield std::unexpected("getting cam0 frame: "
                                         + frame.error());
            }

            reading->frame[0] = frame.value();
            curr_cam0++;
        }

        if (cam1_ts == min) {
            auto frame = getLocalFrame(curr_cam1->path, curr_cam1->timestamp);
            if (!frame) {
                spdlog::error(
                    LOG_ID " could not get frame for cam1 at ts:{}, path:{}",
                    curr_cam1->timestamp, std::string(curr_cam1->path));

                co_yield std::unexpected("getting cam1 frame: "
                                         + frame.error());
            }

            reading->frame[1] = frame.value();
            curr_cam1++;
        }

        co_yield reading;
    }
}
