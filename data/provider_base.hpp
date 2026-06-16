#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <generator>
#include <memory>
#include <vector>

namespace Eigen {
using VectorX8u = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
};

namespace wslam::data {
struct IMUSensorParams {
    Eigen::Matrix4d T_BS;

    uint32_t rate_hz;

    double gyroscope_noise_density;
    double gyroscope_random_walk;
    double accelerometer_noise_density;
    double accelerometer_random_walk;
};

struct CamSensorParams {
    Eigen::Matrix4d T_BS;

    uint32_t rate_hz;
    std::array<int, 2> resolution;  // [width, height]

    std::string camera_model;
    Eigen::Vector4d intrinsics;  // [fu, fv, cu, cv]

    std::string distortion_model;
    Eigen::Vector4d distortion_coefficients;
};

struct SensorParams {
    IMUSensorParams imu;
    std::vector<CamSensorParams> cams;
};

enum class FrameColor : uint8_t { BW, RGB };

struct PixelRGB {
    uint8_t r, g, b;
};

struct FrameRGB {
    uint64_t timestamp;
    std::vector<PixelRGB> pixels;
    uint16_t width;
    uint16_t height;
};
struct FrameBW {
    uint64_t timestamp;
    std::vector<float> pixels;
    uint16_t width;
    uint16_t height;
};

struct IMUReading {
    uint64_t timestamp;

    // NOLINTBEGIN(readability-magic-numbers)
    Eigen::Matrix<float, 6, 1> vals;
    [[nodiscard]] float wx() const { return vals[0]; }
    [[nodiscard]] float wy() const { return vals[1]; }
    [[nodiscard]] float wz() const { return vals[2]; }
    [[nodiscard]] float ax() const { return vals[3]; }
    [[nodiscard]] float ay() const { return vals[4]; }
    [[nodiscard]] float az() const { return vals[5]; }
    // NOLINTEND(readability-magic-numbers)
};

template <size_t FrameCnt>
struct Reading {
    std::array<std::optional<FrameBW>, FrameCnt> frames;
    std::optional<IMUReading> imu;
};

/**
 * Summarizes IMU readings by interpolating them to match the
 * specific timestamp of an image frame.
 */
IMUReading SummarizeReadings(std::span<IMUReading> readings, uint64_t image_ts);

/**
 * Converts RGB to Grayscale using the standard Luminance formula.
 */
FrameBW FrameRGBToBW(const FrameRGB& frame);

/**
 * Converts Grayscale to RGB by replicating the luminance channel.
 */
FrameRGB FrameBWToRGB(const FrameBW& frame);

std::expected<FrameBW, std::string> getLocalFrame(
    const std::filesystem::path& path, uint64_t timestamp);

template <size_t CamCnt>
class ProviderBase {
   public:
    using ReadingType = std::expected<Reading<CamCnt>, std::string>;

    virtual std::generator<ReadingType> getReadings() = 0;

    virtual std::expected<SensorParams, std::string> getSensorParams() = 0;

    virtual ~ProviderBase() = default;
};

template <size_t Requested, size_t Available>
std::generator<std::expected<Reading<Requested>, std::string>> AdaptProvider(
    std::shared_ptr<ProviderBase<Available>> provider) {
    for (auto src : provider->getReadings()) {
        if (!src) {
            co_yield std::unexpected(src.error());
            continue;
        }

        // Take first N frames
        Reading<Requested> dest;
        for (size_t i = 0; i < Requested; ++i) {
            dest.frames[i] = std::move(src->frames[i]);
        }
        dest.imu = src->imu;

        co_yield dest;
    }
}
};  // namespace wslam::data

struct GroundTruth {
    uint64_t timestamp;
    float tx;
    float ty;
    float tz;
    float qx;
    float qy;
    float qz;
};

namespace std {
using namespace wslam::data;

// Helper formatter for Eigen::Matrix4d
template <>
struct formatter<Eigen::Matrix4d> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static auto format(const Eigen::Matrix4d& mat, std::format_context& ctx) {
        auto out = ctx.out();
        out = std::format_to(out, "\n");
        for (int row = 0; row < 4; ++row) {
            out = std::format_to(
                out, "  [{:10.6f}, {:10.6f}, {:10.6f}, {:10.6f}]", mat(row, 0),
                mat(row, 1), mat(row, 2), mat(row, 3));
            if (row < 3) {
                out = std::format_to(out, "\n");
            }
        }
        return out;
    }
};

// Helper formatter for Eigen::Vector4d
template <>
struct formatter<Eigen::Vector4d> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static auto format(const Eigen::Vector4d& vec, std::format_context& ctx) {
        return std::format_to(ctx.out(), "[{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                              vec(0), vec(1), vec(2), vec(3));
    }
};

template <>
struct formatter<IMUSensorParams> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static auto format(const IMUSensorParams& imu, std::format_context& ctx) {
        return std::format_to(
            ctx.out(),
            "IMUSensorParams {{\n"
            "  T_BS:                       {}\n"
            "  rate_hz:                    {}\n"
            "  gyroscope_noise_density:    {:.6e}\n"
            "  gyroscope_random_walk:      {:.6e}\n"
            "  accelerometer_noise_density:{:.6e}\n"
            "  accelerometer_random_walk:  {:.6e}\n"
            "}}",
            imu.T_BS, imu.rate_hz, imu.gyroscope_noise_density,
            imu.gyroscope_random_walk, imu.accelerometer_noise_density,
            imu.accelerometer_random_walk);
    }
};

template <>
struct formatter<CamSensorParams> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static auto format(const CamSensorParams& cam, std::format_context& ctx) {
        return std::format_to(ctx.out(),
                              "CamSensorParams {{\n"
                              "  T_BS:                  {}\n"
                              "  rate_hz:               {}\n"
                              "  resolution:            [{}x{}]\n"
                              "  camera_model:          {}\n"
                              "  intrinsics:            {}\n"
                              "  distortion_model:      {}\n"
                              "  distortion_coefficients: {}\n"
                              "}}",
                              cam.T_BS, cam.rate_hz, cam.resolution[0],
                              cam.resolution[1], cam.camera_model,
                              cam.intrinsics, cam.distortion_model,
                              cam.distortion_coefficients);
    }
};

template <>
struct formatter<SensorParams> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static auto format(const SensorParams& sensor, std::format_context& ctx) {
        auto out = ctx.out();
        out = std::format_to(out, "SensorParams {{\n  imu: {}\n  cams ({}):\n",
                             sensor.imu, sensor.cams.size());
        for (size_t i = 0; i < sensor.cams.size(); ++i) {
            out = std::format_to(out, "    [{}]: {}\n", i, sensor.cams[i]);
        }
        return std::format_to(out, "}}");
    }
};
};  // namespace std
