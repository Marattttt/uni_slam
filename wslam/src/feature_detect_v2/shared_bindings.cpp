#include "shared_bindings.hpp"

#include <spdlog/spdlog.h>

#include <format>

#include "raco_lightglue.hpp"

using namespace wslam::v2;

#define LOG_ID "[Shared Bindings v2]"

std::optional<std::string> SharedBindings::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = initTextures()) {
        return "textures: " + err.value();
    }

    if (auto err = initBuffers()) {
        return "buffers: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> SharedBindings::initTextures() {
    spdlog::info(LOG_ID " allocating 2 pyramids of {} textures",
                 GPUConst::levels_of_detail);

    auto awaiter = gpu_->getAwaiter();

    for (const auto slot : {PyramidSlot::Current, PyramidSlot::Previous}) {
        for (uint32_t lod = 0; lod < GPUConst::levels_of_detail; ++lod) {
            initTexture(awaiter, slot, lod);
        }
    }

    return awaiter.executeAll();
}

void SharedBindings::initTexture(compute::Awaiter& awaiter, PyramidSlot slot,
                                 uint32_t lod) {
    const auto [width, height] = GPUBindingSize::getPyramidLayerDimensions(
        {static_cast<uint8_t>(lod)});

    const std::string label = std::format(
        "{} frame texture for lod {}",
        slot == PyramidSlot::Current ? "Current" : "Previous", lod);
    const wgpu::TextureDescriptor texture_desc{
        .label = label.c_str(),
        .usage = wgpu::TextureUsage::TextureBinding
                 | wgpu::TextureUsage::StorageBinding
                 | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = static_cast<uint32_t>(width),
                 .height = static_cast<uint32_t>(height),
                 .depthOrArrayLayers = 1,},
        .format = wgpu::TextureFormat::R32Float,
    };

    auto create_texture = [&] {
        pyramids_.at(static_cast<size_t>(slot)).at(lod)
            = gpu_->getDevice().CreateTexture(&texture_desc);
    };

    awaiter.runChecked(create_texture, LOG_ID " create texture");
}

std::optional<std::string> SharedBindings::initBuffers() {
    using BU = wgpu::BufferUsage;
    constexpr auto kUsage = BU::Storage | BU::CopySrc | BU::CopyDst;

    auto images = gpu_->createDedicatedBuffer(raco_lightglue::kImagesBytes,
                                              kUsage, "Model images");
    if (!images) {
        return "images: " + images.error();
    }
    images_buffer_ = std::move(images).value();

    auto keypoints = gpu_->createDedicatedBuffer(
        raco_lightglue::KeypointsBytes(keypoint_count_), kUsage, "Model keypoints");
    if (!keypoints) {
        return "keypoints: " + keypoints.error();
    }
    keypoints_buffer_ = std::move(keypoints).value();

    return std::nullopt;
}
