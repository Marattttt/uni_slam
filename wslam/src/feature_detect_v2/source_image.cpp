#include "source_image.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <ranges>
#include <vector>

#include "common.hpp"

using namespace wslam;

#define LOG_ID "[Source Image pass]"

std::string v2::SourceImagePass::getId() const { return LOG_ID; }

std::optional<std::string> v2::SourceImagePass::initialize() {
    spdlog::info(LOG_ID " initializing");

    return std::nullopt;
}

std::optional<std::string> v2::SourceImagePass::submitWork() {
    spdlog::info(LOG_ID " Executing");

    // A queue write, so the passes submitted after this one see the new frame
    if (auto err = writeSource()) {
        return "writing source frame: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> v2::SourceImagePass::writeSource() {
    spdlog::info(LOG_ID " Writing source texture");

    wgpu::Queue queue = gpu_->getDevice().GetQueue();

    wgpu::TexelCopyTextureInfo copy_info{
        .texture = bindings_->getSourceTexture(),
        .mipLevel = 0,
        .origin = {.x = 0, .y = 0},
        .aspect = wgpu::TextureAspect::All,
    };

    wgpu::TexelCopyBufferLayout copy_layout{
        .offset = 0,
        .bytesPerRow = GPUConst::pixel_size * GPUConst::frame_width,
        .rowsPerImage = GPUConst::frame_height,
    };

    wgpu::Extent3D extent{.width = GPUConst::frame_width,
                          .height = GPUConst::frame_height,
                          .depthOrArrayLayers = 1,};

    auto data = image_getter_();
    if (!data.has_value()) {
        return "end of data";
    }

    if (data->empty()) {
        return "data is empty";
    }

    const auto image_head = data.value() | std::views::take(10)
                            | std::views::transform([](const auto& x) {
                                  return static_cast<uint8_t>(x);
                              })
                            | std::ranges::to<std::vector<uint8_t>>();
    spdlog::debug(LOG_ID " Got image data. Head: {}", image_head);

    auto awaiter = gpu_->getAwaiter();

    auto callback = [&]() {
        queue.WriteTexture(&copy_info, data->data(), data->size_bytes(),
                           &copy_layout, &extent);
    };

    return awaiter.runChecked(callback, LOG_ID " write texture source")
        .executeAll(false)
        .transform([](auto&& err) { return "write source texture: " + err; });
}
