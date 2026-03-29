#include "common.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include "awaiter.hpp"
#include "gpu.hpp"

using namespace wslam;
namespace c = wslam::compute;

#define LOG_ID "[Shared Bindings]"

namespace {
const std::string kConstantsBinding = "constants";
};  // namespace

const c::BufferBinding& GpuSharedBindings::getConstantsBinding() const {
    assert(buf_bindings_.size() > 0);
    return buf_bindings_.at(kConstantsBinding);
}

std::optional<std::string> GpuSharedBindings::initialize(
    compute::GPU& gpu,
    std::span<const std::byte, sizeof(GPUConst)> constant_data) {
    assert(group == nullptr);
    assert(layout == nullptr);

    if (auto err = initTexture(gpu)) {
        return "textures: " + err.value();
    }

    if (auto err = initBindGroupLayout(gpu)) {
        return "bind group layout: " + err.value();
    };

    if (auto err = initBindGroup(gpu)) {
        return "bind group: " + err.value();
    }

    if (auto err = writeConstants(gpu, constant_data)) {
        return "wriring constants: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> GpuSharedBindings::initBindGroupLayout(
    const compute::GPU& gpu) {
    // Constants
    layout_entries_[0].binding = 0;
    layout_entries_[0].buffer.type = wgpu::BufferBindingType::Uniform;
    layout_entries_[0].visibility = wgpu::ShaderStage::Compute;

    wgpu::BindGroupLayoutDescriptor desc{
        .label = "Common bindings layout for SLAM",
        .entryCount = static_cast<uint32_t>(layout_entries_.size()),
        .entries = layout_entries_.data(),
    };

    auto awaiter = gpu.getAwaiter();
    awaiter.addCall(
        [=, this]() { layout = gpu.getDevice().CreateBindGroupLayout(&desc); },
        "Create bind group layout");

    return awaiter.executeAll();
}

std::optional<std::string> GpuSharedBindings::initBindGroup(compute::GPU& gpu) {
    spdlog::info("[GPU Shared bindings] initializing bind group");
    assert(layout != nullptr);

    // Constants
    group_entries_[0] = {.binding = 0, .size = GPUBindingSize::constants};

    std::span entries{group_entries_};
    auto bindings = gpu.assignBuffersAndOffsets(
        {{kConstantsBinding, c::GPU::BgBinding{
                                 .buf_type = c::BufferType::Uniform,
                                 .bg_entry = group_entries_[0],
                             }}});

    if (bindings) {
        buf_bindings_ = std::move(bindings.value());
    } else {
        return "assigning buffers: " + bindings.error();
    }

    wgpu::BindGroupDescriptor desc{.label = "Common bindings for SLAM",
                                   .layout = layout,
                                   .entryCount = group_entries_.size(),
                                   .entries = group_entries_.data()};

    auto awaiter = gpu.getAwaiter();
    awaiter.addCall([&]() { group = gpu.getDevice().CreateBindGroup(&desc); },
                    "Bind group for common bindings");

    return awaiter.executeAll();
}

std::optional<std::string> GpuSharedBindings::initTexture(const c::GPU& gpu) {
    auto device = gpu.getDevice();

    wgpu::TextureDescriptor texture_desc{
        .label = "Frame texture",
        .usage = wgpu::TextureUsage::TextureBinding
                 | wgpu::TextureUsage::StorageBinding
                 | wgpu::TextureUsage::CopyDst,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = GPUConst::frame_width,
                 .height = GPUConst::frame_height,
                 .depthOrArrayLayers = 1},
        .format = wgpu::TextureFormat::R32Float,
        .mipLevelCount = GPUConst::pyr_levels,
    };

    auto awaiter = gpu.getAwaiter();

    auto create_texture
        = [&]() { frame_ = device.CreateTexture(&texture_desc); };
    awaiter.addCall(std::move(create_texture), LOG_ID " create texture");

    if (auto err = awaiter.executeAll()) {
        return "create layer texture views " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> GpuSharedBindings::writeConstants(
    compute::GPU& gpu,
    std::span<const std::byte, GPUBindingSize::constants> constant_data) {
    const auto& device = gpu.getDevice();
    const auto queue = device.GetQueue();
    const auto encoder = device.CreateCommandEncoder();

    if (auto err = gpu.fillNonInputBuffer(
            encoder, constant_data, buf_bindings_.at(kConstantsBinding))) {
        return "filling gpu buf: " + err.value();
    };

    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto awaiter = gpu.getAwaiter();

    auto callback = [&]() -> wgpu::Future {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView msg,
               decltype(this) user_data) {
                (void)user_data;
                if (status == wgpu::QueueWorkDoneStatus::Error) {
                    spdlog::error(LOG_ID
                                  " could not fill constants data; msg: {}",
                                  std::string(msg));
                }
            },
            this);
    };
    awaiter.addCall(std::move(callback), "filling constatns buffer");

    if (auto err = awaiter.executeAll()) {
        return "executing commands: " + err.value();
    }

    return std::nullopt;
}
