#include "pack_model_input.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <array>

#include "common.hpp"
#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[Pack Model Input pass]"

namespace {
constexpr uint32_t kWorkgroupSize = 16;

constexpr std::array<wgpu::ConstantEntry, 4> kShaderConstants{
    wgpu::ConstantEntry{
        .key = "OUT_WIDTH",
        .value = model::kInputWidth,
    },
    wgpu::ConstantEntry{
        .key = "OUT_HEIGHT",
        .value = model::kInputHeight,
    },
    wgpu::ConstantEntry{
        .key = "CROP_X",
        .value = model::kCropX,
    },
    wgpu::ConstantEntry{
        .key = "CROP_Y",
        .value = model::kCropY,
    },
};
}  // namespace

std::string v2::PackModelInputPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::PackModelInputPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = initTextureViews()) {
        return "texture views: " + err.value();
    }
    if (auto err = initBindGroupLayout()) {
        return "bind group layout: " + err.value();
    }
    if (auto err = initBindGroup()) {
        return "bind group: " + err.value();
    }
    if (auto err = initComputePipeline()) {
        return "compute pipeline: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> v2::PackModelInputPass::initTextureViews() {
    const wgpu::TextureViewDescriptor desc{
        .label = LOG_ID " model level view",
        .format = wgpu::TextureFormat::R32Float,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .usage = wgpu::TextureUsage::TextureBinding,
    };

    auto create_views = [&] {
        left_view_
            = bindings_->getTexture(PyramidSlot::Previous, model::kInputLod)
                  .CreateView(&desc);
        right_view_
            = bindings_->getTexture(PyramidSlot::Current, model::kInputLod)
                  .CreateView(&desc);
    };

    return gpu_->getAwaiter()
        .runChecked(create_views, LOG_ID " create texture views")
        .executeAll();
}

std::optional<std::string> v2::PackModelInputPass::initBindGroupLayout() {
    // R32Float is only filterable with a device feature, and the shader reads
    // texels with textureLoad, so bind the views as unfilterable
    const std::array<wgpu::BindGroupLayoutEntry, 3> layout_entries{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .texture = {
                .sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Compute,
            .texture = {
                .sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {
                .type = wgpu::BufferBindingType::Storage,
                .minBindingSize = model::kImagesBytes,
            },
        },
    };

    const wgpu::BindGroupLayoutDescriptor desc{
        .label = LOG_ID " bind group layout",
        .entryCount = layout_entries.size(),
        .entries = layout_entries.data(),
    };

    return gpu_->getAwaiter()
        .runChecked(
            [&] {
                bind_group_layout_
                    = gpu_->getDevice().CreateBindGroupLayout(&desc);
            },
            LOG_ID " create bind group layout")
        .executeAll();
}

std::optional<std::string> v2::PackModelInputPass::initBindGroup() {
    const std::array<wgpu::BindGroupEntry, 3> entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = left_view_},
        wgpu::BindGroupEntry{.binding = 1, .textureView = right_view_},
        wgpu::BindGroupEntry{
            .binding = 2,
            .buffer = bindings_->getImagesBuffer(),
            .offset = 0,
            .size = model::kImagesBytes,
        },
    };

    const wgpu::BindGroupDescriptor desc{
        .label = LOG_ID " bind group",
        .layout = bind_group_layout_,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };

    return gpu_->getAwaiter()
        .runChecked(
            [&] { bind_group_ = gpu_->getDevice().CreateBindGroup(&desc); },
            LOG_ID " create bind group")
        .executeAll();
}

std::optional<std::string> v2::PackModelInputPass::initComputePipeline() {
    spdlog::debug(LOG_ID " Initializing compute pipeline");

    auto mod
        = gpu_->loadShaderModule("pack_model_input.wgsl", "pack model input");
    if (!mod) {
        return "creating shader module: " + mod.error();
    }

    auto awaiter = gpu_->getAwaiter();

    const wgpu::PipelineLayoutDescriptor layout_desc{
        .label = LOG_ID " pipeline layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bind_group_layout_,
    };

    wgpu::PipelineLayout layout;
    awaiter.runChecked(
        [&] { layout = gpu_->getDevice().CreatePipelineLayout(&layout_desc); },
        LOG_ID " create pipeline layout");

    const wgpu::ComputePipelineDescriptor desc{
        .label = LOG_ID " compute pipeline",
        .layout = layout,
        .compute = {
            .module = mod.value(),
            .entryPoint = "main",
            .constantCount = kShaderConstants.size(),
            .constants = kShaderConstants.data(),
        },
    };

    awaiter.runChecked(
        [&] {
            compute_pipeline_ = gpu_->getDevice().CreateComputePipeline(&desc);
        },
        LOG_ID " create compute pipeline");

    return awaiter.executeAll();
}

std::optional<std::string> v2::PackModelInputPass::submitWork() {
    if (!bindings_->hasPrevious()) {
        spdlog::info(LOG_ID " No previous pyramid yet; nothing to pack");
        return std::nullopt;
    }

    spdlog::info(LOG_ID " Executing");

    const uint32_t groups_x = CeilDiv(model::kInputWidth, kWorkgroupSize);
    const uint32_t groups_y = CeilDiv(model::kInputHeight, kWorkgroupSize);

    return gpu_->submitCommands(
        LOG_ID " encoder", [&](const wgpu::CommandEncoder& encoder) {
            auto pack = [&] {
                const wgpu::ComputePassDescriptor pass_desc{
                    .label = LOG_ID " compute pass",
                };
                const wgpu::ComputePassEncoder pass
                    = encoder.BeginComputePass(&pass_desc);

                pass.SetPipeline(compute_pipeline_);
                pass.SetBindGroup(0, bind_group_);
                pass.DispatchWorkgroups(groups_x, groups_y);

                pass.End();
            };

            return gpu_->getAwaiter()
                .runChecked(pack, LOG_ID " pack model input")
                .executeAll()
                .transform([](const auto& err) { return "awaiter: " + err; });
        });
}
