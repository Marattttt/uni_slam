#include "cpu_model_input.hpp"

#include <cstddef>
#include <cstdint>
#include <format>

#include "data/provider_base.hpp"
#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

v2::CpuModelInput::CpuModelInput()
    : values_(model::kImagesBytes / sizeof(float)) {}

std::optional<std::string> v2::CpuModelInput::writeFrame(
    const data::FrameBW& frame) {
    const auto width = static_cast<uint32_t>(frame.width);
    const auto height = static_cast<uint32_t>(frame.height);

    if (width != model::kSourceWidth || height != model::kSourceHeight) {
        return std::format(
            "frame is {}x{}, the model input is cropped from {}x{}", width,
            height, model::kSourceWidth, model::kSourceHeight);
    }

    const size_t pixels = static_cast<size_t>(width) * height;
    if (frame.pixels.size() != pixels) {
        return std::format("frame has {} pixels, expected {}",
                           frame.pixels.size(), pixels);
    }

    constexpr size_t kPlane
        = static_cast<size_t>(model::kInputWidth) * model::kInputHeight;

    for (uint32_t y = 0; y < model::kInputHeight; ++y) {
        const float* source = frame.pixels.data()
                              + (static_cast<size_t>(y) + model::kCropY) * width
                              + model::kCropX;
        float* destination
            = values_.data() + static_cast<size_t>(y) * model::kInputWidth;
        for (uint32_t x = 0; x < model::kInputWidth; ++x) {
            const float gray = source[x];
            destination[x] = gray;
            destination[x + kPlane] = gray;
            destination[x + 2 * kPlane] = gray;
        }
    }

    return std::nullopt;
}
