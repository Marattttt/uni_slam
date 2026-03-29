#include "detect_corners.hpp"

#include <spdlog/spdlog.h>
#include <unistd.h>
#include <webgpu/webgpu_cpp.h>

#include "awaiter.hpp"

using namespace wslam;
using namespace std::chrono_literals;

namespace c = wslam::compute;

#define PASS_ID "[Pass Detect Corners]"

namespace {
const std::string kOutputBinding = "output";
};

std::string PassDetectCorners::getId() const { return PASS_ID; }

std::optional<std::string> PassDetectCorners::initialize() {
    spdlog::info(getId() + " initializing");

    assert(gpu_->getDevice() != nullptr);
    assert(gpu_->getInstance() != nullptr);

    if (auto err = initBindGroupLayout()) {
        return "bind group layout: " + err.value();
    }

    if (auto err = initBindGroup()) {
        return "bindgroup: " + err.value();
    }

    if (auto err = initComputePipeline()) {
        return "compute pipeline: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> PassDetectCorners::initBindGroupLayout() {
    spdlog::info(getId() + " initializing bind group layout");

    const std::vector<wgpu::BindGroupLayoutEntry> bindings{
        {

            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        },
    };

    wgpu::BindGroupLayoutDescriptor desc{
        .label = PASS_ID " Bind group layout",
        .entryCount = bindings.size(),
        .entries = bindings.data(),
    };

    auto awaiter = gpu_->getAwaiter();

    awaiter.addCall([&]() { gpu_->getDevice().CreateBindGroupLayout(&desc); },
                    getId() + " create bind group layout");

    return awaiter.executeAll();
}

std::optional<std::string> PassDetectCorners::initBindGroup() {
    spdlog::info(getId() + " initializing bind group");

    std::array<wgpu::BindGroupEntry, 1> bindings{
        wgpu::BindGroupEntry{.binding = 0, .size = 1},
    };

    auto binding = gpu_->assignBuffersAndOffsets({{
        kOutputBinding,
        c::GPU::BgBinding{
            .buf_type = c::BufferType::StorageA,
            .bg_entry = bindings.front(),
        },
    }});

    if (binding) {
        buf_bindings_ = std::move(binding.value());
    } else {
        return "assigning offsets: " + binding.error();
    }

    wgpu::BindGroupDescriptor desc{.label = PASS_ID " bind group",
                                   .layout = bind_group_layout_,
                                   .entryCount = bindings.size(),
                                   .entries = bindings.data()};

    auto awaiter = gpu_->getAwaiter();

    awaiter.addCall(
        [&]() { bind_group_ = gpu_->getDevice().CreateBindGroup(&desc); },
        getId() + " create bind group");

    return awaiter.executeAll();
}

std::optional<std::string> PassDetectCorners::initComputePipeline() {
    spdlog::info(getId() + " initializing compute pipeline");

    wgpu::ShaderModule module;

    auto mod_err
        = gpu_->loadShaderModule("detect_corners.wgsl", "detect corners")
              .transform([&module](auto mod) { module = std::move(mod); });

    if (!mod_err) {
        return "loading shader module: " + mod_err.error();
    }

    std::vector<wgpu::BindGroupLayout> layouts{shared_bindings_.layout,
                                               bind_group_layout_};

    wgpu::PipelineLayoutDescriptor pipeline_desc{
        .label = PASS_ID " pipeline layout",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
    };

    auto awaiter = gpu_->getAwaiter();

    awaiter.addCall(
        [&]() {
            compute_pipeline_layout_
                = gpu_->getDevice().CreatePipelineLayout(&pipeline_desc);
        },
        getId() + " create pipeline layout");

    if (auto err = awaiter.executeAll()) {
        return "pipeline layout: " + err.value();
    }

    wgpu::ComputePipelineDescriptor compute_desc{
        .label = PASS_ID " compute pipeline",
        .layout = compute_pipeline_layout_,
        .compute = {.module = module, .entryPoint = "detectCorners"}};

    awaiter.addCall(
        [&]() {
            gpu_->getDevice().CreateComputePipeline(&compute_desc);
            std::this_thread::sleep_for(500ms);
        },
        getId() + " create compute pipeline");

    if (auto err = awaiter.executeAll()) {
        return err;
    }

    return std::nullopt;
}

std::optional<std::string> PassDetectCorners::execute() {
    const wgpu::Device device = gpu_->getDevice();
    const wgpu::Queue queue = device.GetQueue();
}
