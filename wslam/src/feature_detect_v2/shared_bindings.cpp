#include "shared_bindings.hpp"

#include <spdlog/spdlog.h>

#include "common.hpp"
#include "raco_lightglue.hpp"

using namespace wslam::v2;

#define LOG_ID "[Shared Bindings v2]"

std::optional<std::string> SharedBindings::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = initTexture()) {
        return "texture: " + err.value();
    }

    if (auto err = initBuffer()) {
        return "buffer: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> SharedBindings::initTexture() {
    spdlog::info(LOG_ID " allocating the source texture ({}x{})",
                 GPUConst::frame_width, GPUConst::frame_height);

    const wgpu::TextureDescriptor texture_desc{
        .label = "Frame texture",
        .usage = wgpu::TextureUsage::TextureBinding
                 | wgpu::TextureUsage::CopyDst,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = GPUConst::frame_width,
                 .height = GPUConst::frame_height,
                 .depthOrArrayLayers = 1,},
        .format = wgpu::TextureFormat::R32Float,
    };

    return gpu_->getAwaiter()
        .runChecked(
            [&] {
                source_texture_
                    = gpu_->getDevice().CreateTexture(&texture_desc);
            },
            LOG_ID " create texture")
        .executeAll();
}

std::optional<std::string> SharedBindings::initBuffer() {
    using BU = wgpu::BufferUsage;
    constexpr auto kUsage = BU::Storage | BU::CopySrc | BU::CopyDst;

    auto images = gpu_->createDedicatedBuffer(raco_lightglue::kImagesBytes,
                                              kUsage, "Model images");
    if (!images) {
        return "images: " + images.error();
    }
    images_buffer_ = std::move(images).value();

    return std::nullopt;
}
