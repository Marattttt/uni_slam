// The CPU execution path's host model input (feature_detect_v2/
// cpu_model_input.hpp) and its wiring through the v2 SensorLoaderPass. No
// GPU or dataset needed

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <generator>
#include <memory>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "common.hpp"
#include "data/provider_base.hpp"
#include "feature_detect_v2/cpu_model_input.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"
#include "feature_detect_v2/sensor_loader.hpp"

using namespace wslam;
namespace model = v2::raco_lightglue;
using ReadingType = std::expected<data::Reading<1>, std::string>;

namespace {

// Distinct values that are exact in float, so the pack must copy the same bits
std::vector<float> SyntheticPixels() {
    std::vector<float> pixels(static_cast<size_t>(model::kSourceWidth)
                              * model::kSourceHeight);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<float>(i % 4096) / 4096.0F;
    }
    return pixels;
}

data::FrameBW SyntheticFrame() {
    return data::FrameBW{
        .timestamp = 42,
        .pixels = SyntheticPixels(),
        .width = static_cast<uint16_t>(model::kSourceWidth),
        .height = static_cast<uint16_t>(model::kSourceHeight),
    };
}

// One reading with a single frame; the frame's pixels are only copied when
// the loader calls load()
std::generator<ReadingType> OneFrame(const data::FrameBW& frame) {
    data::Reading<1> reading;
    reading.frames[0] = data::PendingFrame{
        .timestamp = frame.timestamp,
        .load = [&frame]() -> std::expected<data::FrameBW, std::string> {
            return frame;
        },
    };
    co_yield reading;
}

constexpr size_t kPlane
    = static_cast<size_t>(model::kInputWidth) * model::kInputHeight;

// Index of the source pixel the model input's (0, 0) comes from
constexpr size_t kFirstSourcePixel
    = static_cast<size_t>(model::kCropY) * model::kSourceWidth + model::kCropX;

bool AllZero(std::span<const float> values) {
    return std::ranges::all_of(values, [](float value) {
        return value == 0.0F;
    });
}

}  // namespace

TEST(CpuModelInputTest, PacksCentreCropIntoThreeIdenticalPlanes) {
    const auto frame = SyntheticFrame();
    v2::CpuModelInput input;

    ASSERT_EQ(input.writeFrame(frame), std::nullopt);

    const auto values = input.values();
    ASSERT_EQ(values.size(), 3 * kPlane);

    for (uint32_t y = 0; y < model::kInputHeight; ++y) {
        for (uint32_t x = 0; x < model::kInputWidth; ++x) {
            const size_t source
                = (static_cast<size_t>(y) + model::kCropY) * model::kSourceWidth
                  + model::kCropX + x;
            const size_t pixel = static_cast<size_t>(y) * model::kInputWidth + x;
            EXPECT_EQ(values[pixel], frame.pixels[source])
                << "channel 0 at (" << x << ", " << y << ")";
            EXPECT_EQ(values[kPlane + pixel], frame.pixels[source])
                << "channel 1 at (" << x << ", " << y << ")";
            EXPECT_EQ(values[2 * kPlane + pixel], frame.pixels[source])
                << "channel 2 at (" << x << ", " << y << ")";
        }
    }

    // The crop's corners
    EXPECT_EQ(values[0], frame.pixels[kFirstSourcePixel]);
    EXPECT_EQ(values[3 * kPlane - 1],
              frame.pixels[(static_cast<size_t>(model::kCropY)
                            + model::kInputHeight - 1)
                               * model::kSourceWidth + model::kCropX
                           + model::kInputWidth - 1]);
}

TEST(CpuModelInputTest, RejectsMismatchedFrame) {
    v2::CpuModelInput input;

    auto wrong_size = SyntheticFrame();
    wrong_size.width = static_cast<uint16_t>(model::kSourceWidth - 1);
    const auto size_error = input.writeFrame(wrong_size);
    ASSERT_TRUE(size_error.has_value());
    EXPECT_TRUE(size_error->contains("cropped from")) << size_error.value();
    EXPECT_EQ(input.values().size(), model::kImagesBytes / sizeof(float));
    EXPECT_TRUE(AllZero(input.values()));

    auto short_pixels = SyntheticFrame();
    short_pixels.pixels.pop_back();
    const auto pixel_error = input.writeFrame(short_pixels);
    ASSERT_TRUE(pixel_error.has_value());
    EXPECT_TRUE(pixel_error->contains("pixels")) << pixel_error.value();
    EXPECT_TRUE(AllZero(input.values()));
}

TEST(CpuModelInputTest, BufferIsStableAcrossFrames) {
    v2::CpuModelInput input;
    const auto first = SyntheticFrame();
    auto second = SyntheticFrame();
    second.pixels[kFirstSourcePixel] = 0.25F;

    ASSERT_EQ(input.writeFrame(first), std::nullopt);
    float* address = input.values().data();
    const size_t size = input.values().size();

    ASSERT_EQ(input.writeFrame(second), std::nullopt);
    // No reallocation: the Ort tensor points into the same buffer
    EXPECT_EQ(input.values().data(), address);
    EXPECT_EQ(input.values().size(), size);
    EXPECT_EQ(input.values()[0], 0.25F);
    EXPECT_EQ(input.values()[kPlane], 0.25F);
    EXPECT_EQ(input.values()[2 * kPlane], 0.25F);
    EXPECT_NE(input.values()[1], 0.25F);
}

TEST(CpuModelInputLoaderTest, FillsThePublishedBuffer) {
    AnyBag storage;
    const auto frame = SyntheticFrame();
    auto input = std::make_shared<v2::CpuModelInput>();
    storage.set(ResourceIdentifier::CpuModelInputName, input);

    v2::SensorLoaderPass loader(OneFrame(frame), storage);
    ASSERT_EQ(loader.initialize(), std::nullopt);
    ASSERT_EQ(loader.execute(), std::nullopt);

    ASSERT_EQ(input->values().size(), 3 * kPlane);
    EXPECT_EQ(input->values()[0], frame.pixels[kFirstSourcePixel]);
    EXPECT_EQ(input->values()[kPlane], frame.pixels[kFirstSourcePixel]);

    // The frame is still published for the loop closure and the recorder
    const auto published = storage.get<data::FrameBW>(
        ResourceIdentifier::GetFrameName(0));
    ASSERT_TRUE(published.has_value());
    EXPECT_EQ(published->timestamp, frame.timestamp);
}

TEST(CpuModelInputLoaderTest, WithoutTheBufferTheLoaderIsUnchanged) {
    AnyBag storage;
    const auto frame = SyntheticFrame();

    v2::SensorLoaderPass loader(OneFrame(frame), storage);
    ASSERT_EQ(loader.initialize(), std::nullopt);
    ASSERT_EQ(loader.execute(), std::nullopt);

    const auto published = storage.get<data::FrameBW>(
        ResourceIdentifier::GetFrameName(0));
    ASSERT_TRUE(published.has_value());
    EXPECT_EQ(published->timestamp, frame.timestamp);
}
