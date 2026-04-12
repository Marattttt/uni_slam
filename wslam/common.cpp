#include "common.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <cmath>

#include "awaiter.hpp"

using namespace wslam;
namespace c = wslam::compute;

#define LOG_ID "[Shared Bindings]"

const wgpu::Texture& GpuSharedBindings::getTexture(size_t lod) const {
    return textures_.at(lod);
}

std::optional<std::string> GpuSharedBindings::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = initTextures()) {
        return "textures: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> GpuSharedBindings::initTextures() {
    spdlog::info(LOG_ID " allocating {} textures for LoDs",
                 GPUConst::levels_of_detail);

    auto awaiter = gpu_->getAwaiter();

    for (uint32_t i = 0; i < GPUConst::levels_of_detail; ++i) {
        initTexture(awaiter, i);
    }

    return awaiter.executeAll();
}

void GpuSharedBindings::initTexture(compute::Awaiter& awaiter, uint32_t lod) {
    const auto device = gpu_->getDevice();

    const double factor = lod == 0 ? 1
                                   : std::pow(GPUConst::lod_scale_factor,
                                              static_cast<double>(lod));
    const auto width = static_cast<uint32_t>(GPUConst::frame_width / factor);
    const auto height = static_cast<uint32_t>(GPUConst::frame_height / factor);

    const std::string label = std::format("Frame texture for lod {}", lod);
    const wgpu::TextureDescriptor texture_desc{
        .label = label.c_str(),
        .usage = wgpu::TextureUsage::TextureBinding
                 | wgpu::TextureUsage::StorageBinding
                 | wgpu::TextureUsage::CopyDst
                 | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = width, .height = height, .depthOrArrayLayers = 1,},
        .format = wgpu::TextureFormat::R32Float,
    };

    auto create_texture
        = [&]() { textures_.at(lod) = device.CreateTexture(&texture_desc); };

    awaiter.addCall(std::move(create_texture), LOG_ID " create texture");
}
