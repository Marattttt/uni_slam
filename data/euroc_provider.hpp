#pragma once

#include <filesystem>

#include "provider_base.hpp"

#ifndef EUROC_DIR_ENV
#define EUROC_DIR_ENV "EUROC_DIR"
#endif

namespace wslam::data {
struct EurocProviderOpts {
    std::filesystem::path path;
};

EurocProviderOpts CreateEurocProviderOpts();

class EurocProvider : public ProviderBase<2> {
   public:
    EurocProvider(const EurocProviderOpts& opts)
        : kImuDir(opts.path / "imu0"),
          kCam0Dir(opts.path / "cam0"),
          kCam1Dir(opts.path / "cam1") {}

    ~EurocProvider() override = default;

    std::generator<std::expected<Reading<2>, std::string>>
    getReadingsSynchronized() override;

    [[nodiscard]] std::expected<SensorParams, std::string> getSensorParams()
        override;

   private:
    const std::filesystem::path kImuDir;
    const std::filesystem::path kCam0Dir;
    const std::filesystem::path kCam1Dir;

    [[nodiscard]] std::optional<std::string> collectFramePaths();
    [[nodiscard]] std::optional<std::string> loadIMUData();
    [[nodiscard]] std::expected<std::array<FrameBW, 2>, std::string>
    getFramesByTs(uint64_t timestamp) const;

    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] std::expected<Reading<2>, std::string> getReading(size_t idx);

    std::optional<std::string> updateIMUSensorParams();
    std::optional<std::string> updateCamSensorParams();

    uint64_t last_timestamp_ = 0;

    std::unique_ptr<IMUSensorParams> imu_params_ = nullptr;
    std::unique_ptr<std::array<CamSensorParams, 2>> cam_params_ = nullptr;

    std::vector<std::pair<uint64_t, std::filesystem::path>> cam0_paths_;
    std::vector<std::pair<uint64_t, std::filesystem::path>> cam1_paths_;
    std::vector<uint8_t> frame_buf_;
    std::vector<IMUReading> imu_readings_;
};
};  // namespace wslam::data
