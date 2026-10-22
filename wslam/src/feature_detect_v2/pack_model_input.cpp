#include "pack_model_input.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <ranges>
#include <vector>

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
        .label = LOG_ID " source view",
        .format = wgpu::TextureFormat::R32Float,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .usage = wgpu::TextureUsage::TextureBinding,
    };

    auto create_views = [&] {
        view_ = bindings_->getSourceTexture().CreateView(&desc);
    };

    return gpu_->getAwaiter()
        .runChecked(create_views, LOG_ID " create texture views")
        .executeAll();
}

std::optional<std::string> v2::PackModelInputPass::initBindGroupLayout() {
    // R32Float is only filterable with a device feature, and the shader reads
    // texels with textureLoad, so bind the views as unfilterable
    const std::array<wgpu::BindGroupLayoutEntry, 2> layout_entries{
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
    const std::array<wgpu::BindGroupEntry, 2> entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = view_},
        wgpu::BindGroupEntry{
            .binding = 1,
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
    spdlog::info(LOG_ID " Executing");

    // The queue write runs before the pack submitted below, so the compute
    // pass sees this frame
    if (auto err = writeSource()) {
        return "writing source frame: " + err.value();
    }

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

std::optional<std::string> v2::PackModelInputPass::writeSource() {
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
