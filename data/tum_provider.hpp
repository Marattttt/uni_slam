#pragma once

#include <expected>
#include <filesystem>
#include <stdexcept>

#include "provider_base.hpp"

namespace wslam::data {
struct TumProviderOpts {
    std::filesystem::path path;
};

class TumProvider : public ProviderBase<1> {
   public:
    TumProvider(const TumProviderOpts& opts)
        : kFramesDir(opts.path / "rgb"), kImuPath(opts.path / "accelerometer") {
        throw std::logic_error("Class not fully implemented");
    }

    // Reading of a new chunk of data may happen, so performance
    // may vary from reading cached data to accessing filesystem
    std::generator<std::expected<Reading<1>, std::string>>
    getReadingsSynchronized() override;

   private:
    static const size_t kIMUBufSize = 30UZ * (1 << 10);  // 30 kilobytes
    static constexpr const std::string kEOF = "eof";
    const std::filesystem::path kFramesDir;
    const std::filesystem::path kImuPath;

    uint32_t frame_width_;
    uint32_t frame_height_;

    std::optional<std::string> loadNextImage();
    std::optional<std::string> collectFramePaths();
    std::optional<std::string> loadIMUData();

    std::vector<std::pair<uint64_t, std::filesystem::path>> frame_paths_;
    std::vector<uint8_t> frame_buf_;
    std::vector<IMUReading> imu_readings;

    uint64_t last_timestamp_ = 0;
};
};  // namespace wslam::data
