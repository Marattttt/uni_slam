#include "data/uzh_fpv_provider.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#include "data/provider_base.hpp"

using namespace wslam::data;
namespace fs = std::filesystem;

#define LOG_ID "[UZH-FPV provider]"

std::expected<uint64_t, std::string> wslam::data::ParseSecondsToNanoseconds(
    std::string_view text) {
    constexpr uint64_t kNanosecondsPerSecond = 1'000'000'000;
    constexpr size_t kFractionDigits = 9;

    const size_t dot = text.find('.');
    const std::string_view whole = text.substr(0, dot);
    const std::string_view fraction = dot == std::string_view::npos
                                          ? std::string_view{}
                                          : text.substr(dot + 1);

    const auto all_digits = [](std::string_view digits) {
        return std::ranges::all_of(digits,
                                   [](char c) { return c >= '0' && c <= '9'; });
    };
    if (whole.empty() || !all_digits(whole) || !all_digits(fraction)) {
        return std::unexpected(std::format("malformed timestamp '{}'", text));
    }

    uint64_t seconds = 0;
    std::from_chars(whole.data(), whole.data() + whole.size(), seconds);

    // Digits past the nanosecond are truncated
    uint64_t nanoseconds = 0;
    for (size_t i = 0; i < kFractionDigits; ++i) {
        nanoseconds
            = nanoseconds * 10
              + (i < fraction.size() ? static_cast<uint64_t>(fraction[i] - '0')
                                     : 0);
    }

    if (seconds > (std::numeric_limits<uint64_t>::max() - nanoseconds)
                      / kNanosecondsPerSecond) {
        return std::unexpected(std::format("timestamp '{}' overflows", text));
    }
    return seconds * kNanosecondsPerSecond + nanoseconds;
}

UzhFpvProviderOpts wslam::data::CreateUzhFpvProviderOpts() {
    UzhFpvProviderOpts opts;
    const auto read = [](const char* name) -> fs::path {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            spdlog::warn(LOG_ID " Env variable {} empty or unset. Using '.'",
                         name);
            return ".";
        }
        return value;
    };
    opts.sequence_dir = read(UZH_FPV_DIR_ENV);
    opts.calibration_dir = read(UZH_FPV_CALIB_DIR_ENV);

    const char* shift = std::getenv("UZH_FPV_CAMERA_TIME_SHIFT_S");
    if (shift != nullptr && shift[0] != '\0') {
        try {
            opts.camera_time_shift_s = std::stod(shift);
        } catch (const std::exception&) {
            spdlog::warn(
                LOG_ID " Ignoring non-numeric UZH_FPV_CAMERA_TIME_SHIFT_S='{}'",
                shift);
        }
    }
    return opts;
}

namespace {
// Data lines of a UZH-FPV text file, split on whitespace. Lines starting with
// '#' are comments
std::expected<std::vector<std::vector<std::string>>, std::string> ReadRows(
    const fs::path& path, size_t columns) {
    std::ifstream file(path);
    if (!file) {
        return std::unexpected(std::format("can't open {}", path.string()));
    }

    std::vector<std::vector<std::string>> rows;
    std::string line;
    size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::vector<std::string> fields;
        std::string field;
        while (stream >> field) {
            fields.push_back(std::move(field));
        }
        if (fields.empty()) {
            continue;
        }
        if (fields.size() != columns) {
            return std::unexpected(
                std::format("{}:{}: expected {} columns, got {}", path.string(),
                            line_number, columns, fields.size()));
        }
        rows.push_back(std::move(fields));
    }
    return rows;
}

// left_images.txt / right_images.txt: "id timestamp img/image_<c>_<id>.png"
std::expected<std::vector<UzhFpvProvider::FrameInfo>, std::string> ReadFrames(
    const fs::path& sequence_dir, std::string_view list_name,
    int64_t time_shift_ns) {
    constexpr size_t kColumns = 3;
    auto rows = ReadRows(sequence_dir / list_name, kColumns);
    if (!rows) {
        return std::unexpected(std::move(rows).error());
    }

    std::vector<UzhFpvProvider::FrameInfo> frames;
    frames.reserve(rows->size());
    for (const auto& row : *rows) {
        auto timestamp = ParseSecondsToNanoseconds(row[1]);
        if (!timestamp) {
            return std::unexpected(
                std::format("{}: {}", list_name, timestamp.error()));
        }
        frames.push_back({
            .timestamp = static_cast<uint64_t>(static_cast<int64_t>(*timestamp)
                                               + time_shift_ns),
            .path = sequence_dir / row[2],
        });
    }
    std::ranges::sort(frames, std::less{},
                      &UzhFpvProvider::FrameInfo::timestamp);

    spdlog::info(LOG_ID " found {} frames in {}", frames.size(), list_name);
    return frames;
}

// imu.txt: "id timestamp wx wy wz ax ay az"
std::expected<std::vector<IMUReading>, std::string> ReadImu(
    const fs::path& path) {
    constexpr size_t kColumns = 8;
    auto rows = ReadRows(path, kColumns);
    if (!rows) {
        return std::unexpected(std::move(rows).error());
    }

    std::vector<IMUReading> readings;
    readings.reserve(rows->size());
    for (const auto& row : *rows) {
        auto timestamp = ParseSecondsToNanoseconds(row[1]);
        if (!timestamp) {
            return std::unexpected(
                std::format("imu.txt: {}", timestamp.error()));
        }
        IMUReading reading{.timestamp = *timestamp, .vals = {}};
        // NOLINTNEXTLINE(readability-magic-numbers)
        for (size_t i = 0; i < 6; ++i) {
            try {
                reading.vals[static_cast<Eigen::Index>(i)]
                    = std::stof(row[2 + i]);
            } catch (const std::exception& err) {
                return std::unexpected(
                    std::format("imu.txt: malformed value '{}': {}", row[2 + i],
                                err.what()));
            }
        }
        readings.push_back(reading);
    }
    std::ranges::sort(readings, std::less{}, &IMUReading::timestamp);
    return readings;
}

// The only file in `dir` whose name starts with `prefix` and ends in .yaml
std::expected<fs::path, std::string> FindYaml(const fs::path& dir,
                                              std::string_view prefix) {
    std::vector<fs::path> found;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(dir, error)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.starts_with(prefix)
            && entry.path().extension() == ".yaml") {
            found.push_back(entry.path());
        }
    }
    if (error) {
        return std::unexpected(
            std::format("listing {}: {}", dir.string(), error.message()));
    }
    if (found.size() != 1) {
        return std::unexpected(
            std::format("expected one {}*.yaml in {}, found {}", prefix,
                        dir.string(), found.size()));
    }
    return found.front();
}

Eigen::Matrix4d ParseMatrixRows(const YAML::Node& node) {
    const auto rows = node.as<std::vector<std::vector<double>>>();
    if (rows.size() != 4 || std::ranges::any_of(rows, [](const auto& row) {
            return row.size() != 4;
        })) {
        throw std::runtime_error("expected a 4x4 matrix");
    }
    Eigen::Matrix4d matrix;
    for (Eigen::Index r = 0; r < 4; ++r) {
        for (Eigen::Index c = 0; c < 4; ++c) {
            matrix(r, c) = rows[static_cast<size_t>(r)][static_cast<size_t>(c)];
        }
    }
    return matrix;
}

Eigen::Vector4d ParseVector4(const YAML::Node& node) {
    const auto values = node.as<std::array<double, 4>>();
    return {values[0], values[1], values[2], values[3]};
}

// Frames per second from the median frame interval
uint32_t FrameRate(const std::vector<UzhFpvProvider::FrameInfo>& frames) {
    if (frames.size() < 2) {
        return 0;
    }
    std::vector<uint64_t> intervals;
    intervals.reserve(frames.size() - 1);
    for (size_t i = 1; i < frames.size(); ++i) {
        intervals.push_back(frames[i].timestamp - frames[i - 1].timestamp);
    }
    const auto middle
        = intervals.begin() + static_cast<std::ptrdiff_t>(intervals.size() / 2);
    std::ranges::nth_element(intervals, middle);
    const auto median = static_cast<double>(*middle);
    return median > 0 ? static_cast<uint32_t>(std::lround(1e9 / median)) : 0;
}

PendingFrame MakePendingFrame(const UzhFpvProvider::FrameInfo& info,
                              std::string_view camera) {
    return PendingFrame{
        .timestamp = info.timestamp,
        .load = [path = info.path, timestamp = info.timestamp,
                 camera = std::string(
                     camera)]() -> std::expected<FrameBW, std::string> {
            auto frame = getLocalFrame(path, timestamp);
            if (!frame) {
                return std::unexpected(std::format(
                    "getting {} frame at ts:{}, path:{}: {}", camera, timestamp,
                    path.string(), std::move(frame).error()));
            }
            return frame;
        },
    };
}
}  // namespace

std::optional<std::string> UzhFpvProvider::loadCalibration() {
    auto camchain = FindYaml(opts_.calibration_dir, "camchain-imucam-");
    if (!camchain) {
        return std::move(camchain).error();
    }
    const fs::path imu_path = opts_.calibration_dir / "imu.yaml";

    SensorParams params;
    try {
        const YAML::Node chain = YAML::LoadFile(camchain->string());
        for (const char* name : {"cam0", "cam1"}) {
            const YAML::Node cam = chain[name];
            if (!cam) {
                return std::format("{}: no {}", camchain->string(), name);
            }
            auto distortion = ParseDistortionModel(
                cam["distortion_model"].as<std::string>());
            if (!distortion) {
                return std::format("{} {}: {}", camchain->string(), name,
                                   distortion.error());
            }
            const Eigen::Matrix4d cam_from_imu
                = ParseMatrixRows(cam["T_cam_imu"]);
            params.cams.push_back(CamSensorParams{
                // The IMU is the body: T_BS = T_imu_cam = T_cam_imu^-1
                .T_BS = cam_from_imu.inverse(),
                .rate_hz = 0,
                .resolution = cam["resolution"].as<std::array<int, 2>>(),
                .camera_model = cam["camera_model"].as<std::string>(),
                .intrinsics = ParseVector4(cam["intrinsics"]),
                .distortion_model = *distortion,
                .distortion_coefficients
                = ParseVector4(cam["distortion_coeffs"]),
            });
            if (params.cams.back().camera_model != "pinhole") {
                return std::format("{} {}: unsupported camera model '{}'",
                                   camchain->string(), name,
                                   params.cams.back().camera_model);
            }
        }
        const auto calibrated_shift_s
            = chain["cam0"]["timeshift_cam_imu"].as<double>(0.0);
        const double time_shift_s
            = opts_.camera_time_shift_s.value_or(calibrated_shift_s);
        if (opts_.camera_time_shift_s) {
            spdlog::warn(LOG_ID
                         " Camera time shift {:+.3f} ms replaces the "
                         "calibrated {:+.3f} ms",
                         time_shift_s * 1e3, calibrated_shift_s * 1e3);
        }
        constexpr double kNanosecondsPerSecond = 1e9;
        camera_time_shift_ns_
            = std::llround(time_shift_s * kNanosecondsPerSecond);

        const YAML::Node imu_file = YAML::LoadFile(imu_path.string());
        // The calibration zips' imu.yaml holds the imu0 block without its key
        const YAML::Node imu = imu_file["imu0"] ? imu_file["imu0"] : imu_file;
        if (!imu["update_rate"]) {
            return std::format("{}: no IMU parameters", imu_path.string());
        }
        params.imu = IMUSensorParams{
            .T_BS = Eigen::Matrix4d::Identity(),
            .rate_hz = static_cast<uint32_t>(
                std::lround(imu["update_rate"].as<double>())),
            .gyroscope_noise_density
            = imu["gyroscope_noise_density"].as<double>(),
            .gyroscope_random_walk = imu["gyroscope_random_walk"].as<double>(),
            .accelerometer_noise_density
            = imu["accelerometer_noise_density"].as<double>(),
            .accelerometer_random_walk
            = imu["accelerometer_random_walk"].as<double>(),
        };
        if (const YAML::Node imu_from_body = imu["T_i_b"]) {
            // T_i_b maps body to IMU coordinates; T_BS is IMU -> body
            params.imu.T_BS = ParseMatrixRows(imu_from_body).inverse();
        }
    } catch (const std::exception& err) {
        return std::format("parsing calibration in {}: {}",
                           opts_.calibration_dir.string(), err.what());
    }

    params_ = std::move(params);
    return std::nullopt;
}

std::optional<std::string> UzhFpvProvider::initialize() {
    if (!params_) {
        if (auto err = loadCalibration()) {
            return std::format("calibration: {}", *err);
        }
    }
    if (imu_readings_.empty()) {
        auto readings = ReadImu(opts_.sequence_dir / "imu.txt");
        if (!readings) {
            return std::format("imu: {}", readings.error());
        }
        if (readings->empty()) {
            return std::format("no IMU readings in {}",
                               opts_.sequence_dir.string());
        }
        imu_readings_ = std::move(readings).value();
    }
    if (cam0_frames_.empty() || cam1_frames_.empty()) {
        auto cam0 = ReadFrames(opts_.sequence_dir, "left_images.txt",
                               camera_time_shift_ns_);
        if (!cam0) {
            return std::format("cam0: {}", cam0.error());
        }
        auto cam1 = ReadFrames(opts_.sequence_dir, "right_images.txt",
                               camera_time_shift_ns_);
        if (!cam1) {
            return std::format("cam1: {}", cam1.error());
        }
        if (cam0->empty() || cam1->empty()) {
            return std::format("no frames in {}", opts_.sequence_dir.string());
        }
        cam0_frames_ = std::move(cam0).value();
        cam1_frames_ = std::move(cam1).value();
        params_->cams[0].rate_hz = FrameRate(cam0_frames_);
        params_->cams[1].rate_hz = FrameRate(cam1_frames_);
    }
    return std::nullopt;
}

std::expected<SensorParams, std::string> UzhFpvProvider::getSensorParams() {
    if (auto err = initialize()) {
        return std::unexpected(std::move(err).value());
    }
    return *params_;
}

std::generator<std::expected<Reading<2>, std::string>>
UzhFpvProvider::getReadings() {
    if (auto err = initialize()) {
        co_yield std::unexpected(
            std::format("initializing: {}", std::move(err).value()));
        co_return;
    }

    auto curr_imu = imu_readings_.cbegin();
    auto curr_cam0 = cam0_frames_.cbegin();
    auto curr_cam1 = cam1_frames_.cbegin();

    constexpr auto kEnd = std::numeric_limits<uint64_t>::max();

    while (true) {
        const uint64_t imu_ts
            = curr_imu == imu_readings_.cend() ? kEnd : curr_imu->timestamp;
        const uint64_t cam0_ts
            = curr_cam0 == cam0_frames_.cend() ? kEnd : curr_cam0->timestamp;
        const uint64_t cam1_ts
            = curr_cam1 == cam1_frames_.cend() ? kEnd : curr_cam1->timestamp;

        const uint64_t min = std::min({imu_ts, cam0_ts, cam1_ts});
        if (min == kEnd) {
            co_return;
        }

        ReadingType reading;
        if (imu_ts == min) {
            reading->imu = *curr_imu;
            ++curr_imu;
        }
        if (cam0_ts == min) {
            reading->frames[0] = MakePendingFrame(*curr_cam0, "cam0");
            ++curr_cam0;
        }
        if (cam1_ts == min) {
            reading->frames[1] = MakePendingFrame(*curr_cam1, "cam1");
            ++curr_cam1;
        }

        co_yield reading;
    }
}
