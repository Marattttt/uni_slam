#include "tum_provider.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

#include "stb_image.h"

using namespace wslam::data;

std::generator<std::expected<Reading<1>, std::string>>
TumProvider::getReadingsSynchronized() {
    throw std::logic_error("not implemented");
}

std::optional<std::string> TumProvider::loadNextImage() {
    assert(frame_paths_.size() > 0);

    const auto* last
        = std::ranges::find_if(frame_paths_, [this](const auto& pair) {
              return pair.first >= last_timestamp_;
          }).base();

    if (last == nullptr) {
        return kEOF;
    }

    const std::filesystem::path& path = last->second;

    const std::unique_ptr<FILE, decltype(&fclose)> file(
        fopen(path.string().c_str(), "rb"), &fclose);

    if (file == nullptr) {
        return "could not open file";
    }

    int width;
    int height;

    const bool image_info_success
        = static_cast<bool>(stbi_info(path.c_str(), &width, &height, nullptr));

    if (!image_info_success) {
        return std::format("getting image info: {}", stbi_failure_reason());
    }

    const uint8_t* data
        = stbi_load_from_file(file.get(), &width, &height, nullptr, 1);

    if (data == nullptr) {
        return "could not read image file";
    }

    frame_width_ = static_cast<uint32_t>(width);
    frame_height_ = static_cast<uint32_t>(height);

    size_t count = static_cast<size_t>(frame_width_) * frame_height_;
    frame_buf_.assign(data, data + count);

    return {};
}

std::optional<std::string> TumProvider::collectFramePaths() {
    std::filesystem::directory_iterator files{kFramesDir};

    std::vector<std::pair<std::filesystem::path, std::string>> invalid_paths;

    frame_paths_
        = files
          | std::ranges::views::filter(
              [](const std::filesystem::directory_entry& entry) {
                  return entry.path().extension() == ".png";
              })
          | std::views::transform(
              [&invalid_paths](const std::filesystem::directory_entry& entry) {
                  try {
                      std::string path_copy = entry.path().string();

                      // Remove extension
                      path_copy = path_copy.substr(path_copy.find_last_of('.'));

                      // Remove the decimal (radix) point
                      path_copy.erase(path_copy.find('.'));

                      uint64_t timestamp = std::stoull(path_copy);

                      return std::make_pair(timestamp, entry);
                  } catch (std::exception& err) {
                      invalid_paths.emplace_back(entry.path(), err.what());
                      return std::make_pair(0UL, entry);
                  }
              })
          | std::ranges::to<
              std::vector<std::pair<uint64_t, std::filesystem::path>>>();

    if (!invalid_paths.empty()) {
        std::string errmsg;
        for (const auto& [path, err] : invalid_paths) {
            if (!errmsg.empty()) {
                errmsg += ", ";
            }
            errmsg += std::format("{}: {}", path.string(), err);
        }

        spdlog::warn("[TUM data provider] Errors parsing paths: [{}]", errmsg);
    }

    if (frame_paths_.empty()) {
        return "no frames found at "
               + std::format("{}/*.png", kFramesDir.string());
    }

    std::ranges::sort(frame_paths_);

    return std::nullopt;
}

std::optional<std::string> TumProvider::loadIMUData() {
    std::ifstream file(kImuPath);
    if (!file.is_open()) {
        return std::format("could not open {}", kImuPath.string());
    }

    std::string line;
    // Ignore first three lines of metadata
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    uint32_t line_count = 0;
    while (std::getline(file, line)) {
        // line format:
        // 1305031098.382263 -0.478957 8.010560 -4.166928

        line_count++;

        if (line.empty()) {
            continue;
        }

        IMUReading reading;
        const char* ptr = line.data();
        const char* line_end = line.data() + line.size();

        // Parse seconds (whole part of timestamp)
        uint64_t seconds;
        std::from_chars_result result = std::from_chars(ptr, line_end, seconds);
        if (result.ec != std::errc{}) {
            return std::format("could not parse timestamp on line {}",
                               line_count);
        }

        ptr = result.ptr + 1;

        // Parse microseconds (after the dot)
        uint64_t microseconds;
        result = std::from_chars(ptr, line_end, microseconds);
        if (result.ec != std::errc{}) {
            return std::format("could not parse timestamp on line {}",
                               line_count);
        }

        const uint64_t million = 1'000'000;
        reading.timestamp = seconds * million + microseconds;

        ptr = result.ptr;  // Advance pointer to where the parsing stopped

        // 2. Parse the IMU sensor values (typically 6: 3x accel, 3x gyro)
        // Using std::size assumes reading.vals is a std::array or a raw array
        // like double vals[6]
        for (size_t i = 0; i < std::size(reading.vals); ++i) {
            ++ptr;
            if (ptr >= line_end) {
                return std::format("unexpected end of line at line {}",
                                   line_count);

                result = std::from_chars(
                    ptr, line_end, reading.vals[static_cast<Eigen::Index>(i)]);

                if (result.ec != std::errc{}) {
                    return std::format("could not parse val[{}] on line {}", i,
                                       line_count);
                }
                ptr = result.ptr;
            }

            // TODO: Store the reading in your class member container
            // e.g., imu_readings_.push_back(reading);
        }

        // Return an empty optional to indicate success (no error string)
        return std::nullopt;
    }

    std::unreachable();
}
