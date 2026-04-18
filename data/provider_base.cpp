#include "provider_base.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <algorithm>
#include <cstdio>
#include <memory>

#include "stb_image.h"

namespace wslam::data {
IMUReading interpolateIMU(std::span<IMUReading> readings, uint64_t image_ts) {
    if (readings.empty()) {
        return {};
    }

    // Find the reading closest to or just after the image_ts
    auto lower_bound = std::ranges::lower_bound(readings, image_ts, {},
                                                &IMUReading::timestamp);

    // Edge cases: if the target timestamp is out of bounds or an exact match
    if (lower_bound == readings.end()) {
        return readings.back();
    }
    if (lower_bound == readings.begin() || lower_bound->timestamp == image_ts) {
        return *lower_bound;
    }

    // Linear Interpolation between the previous and the next IMU reading
    const auto& next = *lower_bound;
    const auto& prev = *(lower_bound - 1);

    float factor = static_cast<float>(image_ts - prev.timestamp)
                   / static_cast<float>(next.timestamp - prev.timestamp);

    float accel_x = prev.ax() + factor * (next.ax() - prev.ax());
    float accel_y = prev.ay() + factor * (next.ay() - prev.ay());
    float accel_z = prev.az() + factor * (next.az() - prev.az());

    // IMU does not contain rotation data
    return {.timestamp = image_ts,
            .vals = {accel_x, accel_y, accel_z, 0, 0, 0}};
}

FrameBW frameRGBToBW(const FrameRGB& frame) {
    FrameBW bw_frame;
    bw_frame.timestamp = frame.timestamp;
    bw_frame.width = frame.width;
    bw_frame.height = frame.height;

    // Resize Eigen vector to match the number of pixels
    size_t num_pixels = frame.pixels.size();
    bw_frame.pixels.resize(num_pixels);

    for (size_t i = 0; i < num_pixels; ++i) {
        const auto& pix = frame.pixels[i];

        // NOLINTBEGIN(readability-magic-numbers)
        bw_frame.pixels[i] = 0.2126F * static_cast<float>(pix.r)
                             + 0.7152F * static_cast<float>(pix.g)
                             + 0.0722F * static_cast<float>(pix.b);
        // NOLINTEND(readability-magic-numbers)
    }

    return bw_frame;
}

FrameRGB FrameBWToRGB(const FrameBW& frame) {
    FrameRGB rgb_frame;
    rgb_frame.timestamp = frame.timestamp;
    rgb_frame.width = frame.width;
    rgb_frame.height = frame.height;

    // Resize std::vector to match the number of pixels in the Eigen vector
    const auto num_pixels = static_cast<size_t>(frame.pixels.size());
    rgb_frame.pixels.resize(num_pixels);

    for (size_t i = 0; i < num_pixels; ++i) {
        const auto val = static_cast<uint8_t>(
            frame.pixels[i] * std::numeric_limits<uint8_t>::max());

        rgb_frame.pixels[i] = {.r = val, .g = val, .b = val};
    }

    return rgb_frame;
}

std::expected<FrameBW, std::string> getLocalFrame(
    const std::filesystem::path& path, uint64_t timestamp) {
    FrameBW frame;

    std::unique_ptr<FILE, std::function<void(FILE*)>> file{
        fopen(path.string().c_str(), "r"), [](FILE* file) { fclose(file); }};

    if (file == nullptr) {
        return std::unexpected("could not open file");
    }

    int width;
    int height;
    int channels;

    std::unique_ptr<uint8_t, decltype(stbi_image_free)*> imgdata{
        stbi_load_from_file(file.get(), &width, &height, &channels, 1),
        &stbi_image_free};

    if (imgdata == nullptr) {
        return std::unexpected(
            std::format("loading image: {}", stbi_failure_reason()));
    }

    frame.height = static_cast<uint16_t>(height);
    frame.width = static_cast<uint16_t>(width);
    frame.timestamp = timestamp;

    const std::span img_span{imgdata.get(),
                             static_cast<size_t>(width * height)};

    frame.pixels = img_span | std::views::transform([](uint8_t pix) {
                       return static_cast<float>(pix)
                              / std::numeric_limits<uint8_t>::max();
                   })
                   | std::ranges::to<std::vector<float>>();

    return frame;
}

}  // namespace wslam::data
