#pragma once

#include <filesystem>
#include <optional>

#include "data/provider_base.hpp"

#ifndef UZH_FPV_DIR_ENV
#define UZH_FPV_DIR_ENV "UZH_FPV_DIR"
#endif

#ifndef UZH_FPV_CALIB_DIR_ENV
#define UZH_FPV_CALIB_DIR_ENV "UZH_FPV_CALIB_DIR"
#endif

namespace wslam::data {
struct UzhFpvProviderOpts {
    // A sequence of the UZH-FPV drone racing dataset in its text format (the
    // *_snapdragon_with_gt.zip downloads): left_images.txt, right_images.txt,
    // imu.txt and img/
    std::filesystem::path sequence_dir;

    // The Kalibr calibration of the sequence's sensor setup (e.g.
    // indoor_45_calib_snapdragon/): camchain-imucam-*.yaml for the cameras
    // and imu.yaml for the IMU noise
    std::filesystem::path calibration_dir;

    // Replaces Kalibr's timeshift_cam_imu (seconds; t_imu = t_cam + shift)
    std::optional<double> camera_time_shift_s;
};

// Reads UZH_FPV_DIR, UZH_FPV_CALIB_DIR and, if set, the
// UZH_FPV_CAMERA_TIME_SHIFT_S override
UzhFpvProviderOpts CreateUzhFpvProviderOpts();

// Stereo + IMU readings of a UZH-FPV sequence. The body frame is the IMU's.
//
// Camera timestamps are moved onto the IMU clock with Kalibr's
// timeshift_cam_imu (t_imu = t_cam + shift; about -15 ms on the Snapdragon
// setups). The stereo pair is hardware-triggered, so both cameras take cam0's
// shift and keep sharing one timestamp. The ground truth (groundtruth.txt) is
// on the IMU clock, so frame timestamps can be scored against it directly
class UzhFpvProvider : public ProviderBase<2> {
   public:
    struct FrameInfo {
        uint64_t timestamp;
        std::filesystem::path path;
    };

    explicit UzhFpvProvider(UzhFpvProviderOpts opts) : opts_(std::move(opts)) {}

    ~UzhFpvProvider() override = default;

    std::generator<ReadingType> getReadings() override;

    [[nodiscard]] std::expected<SensorParams, std::string> getSensorParams()
        override;

   private:
    UzhFpvProviderOpts opts_;

    [[nodiscard]] std::optional<std::string> loadCalibration();
    [[nodiscard]] std::optional<std::string> initialize();

    std::optional<SensorParams> params_;
    // Kalibr's timeshift_cam_imu of cam0, in nanoseconds
    int64_t camera_time_shift_ns_ = 0;

    std::vector<FrameInfo> cam0_frames_;
    std::vector<FrameInfo> cam1_frames_;
    std::vector<IMUReading> imu_readings_;
};

// "1545305703.6524107" seconds -> nanoseconds, exactly (no detour through a
// double, which keeps only ~0.2 us at these magnitudes)
std::expected<uint64_t, std::string> ParseSecondsToNanoseconds(
    std::string_view text);
};  // namespace wslam::data
