#include "fill_pyramid.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>

#include "common.hpp"

using namespace wslam;
namespace c = wslam::compute;

#define LOG_ID "[Fill Pyramid pass]"

std::string FillPyramidPass::getId() const { return LOG_ID; }

std::optional<std::string> FillPyramidPass::initialize() {
    spdlog::info(getId() + " initializing");

    if (auto err = initSampler()) {
        return "sampler: " + err.value();
    }
    if (auto err = initTextureViews()) {
        return "texture views: " + err.value();
    }
    if (auto err = initBindGroupLayout()) {
        return "bind group layout: " + err.value();
    }
    if (auto err = initBindGroups()) {
        return "bind group: " + err.value();
    }
    if (auto err = initComputePipeline()) {
        return "compute pipeline: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> FillPyramidPass::initBindGroupLayout() {
    spdlog::info(LOG_ID " initializing bind group layout");

    std::array<wgpu::BindGroupLayoutEntry, 3> layout_entries{
        {wgpu::BindGroupLayoutEntry{
             .binding = 0,
             .visibility = wgpu::ShaderStage::Compute,
             .texture = {.sampleType = wgpu::TextureSampleType::Float,
                         .viewDimension = wgpu::TextureViewDimension::e2D}},
         wgpu::BindGroupLayoutEntry{
             .binding = 1,
             .visibility = wgpu::ShaderStage::Compute,
             .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
         wgpu::BindGroupLayoutEntry{
             .binding = 2,
             .visibility = wgpu::ShaderStage::Compute,
             .storageTexture
             = {.access = wgpu::StorageTextureAccess::WriteOnly,
                .format = wgpu::TextureFormat::R32Float,
                .viewDimension = wgpu::TextureViewDimension::e2D},
         }}};

    wgpu::BindGroupLayoutDescriptor bgl_desc{};
    bgl_desc.label = "Common layout for bind groups";
    bgl_desc.entryCount = layout_entries.size();
    bgl_desc.entries = layout_entries.data();

    auto callback = [&]() {
        bind_group_layout_ = gpu_->getDevice().CreateBindGroupLayout(&bgl_desc);
    };

    auto awaiter = gpu_->getAwaiter();
    awaiter.addCall(std::move(callback), LOG_ID " create bind group");

    return awaiter.executeAll();
}

std::optional<std::string> FillPyramidPass::initSampler() {
    wgpu::SamplerDescriptor sampler_desc{};
    sampler_desc.minFilter = wgpu::FilterMode::Linear;
    sampler_desc.magFilter = wgpu::FilterMode::Linear;
    sampler_desc.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
    sampler_desc.addressModeU = wgpu::AddressMode::ClampToEdge;
    sampler_desc.addressModeV = wgpu::AddressMode::ClampToEdge;

    auto callback
        = [&]() { sampler_ = gpu_->getDevice().CreateSampler(&sampler_desc); };
    auto awaiter = gpu_->getAwaiter();
    awaiter.addCall(std::move(callback), LOG_ID " creating sampler");

    return awaiter.executeAll();
}

std::optional<std::string> FillPyramidPass::initTextureViews() {
    spdlog::info(LOG_ID " initializing {} TextureView pairs",
                 GPUConst::levels_of_detail);

    auto awaiter = gpu_->getAwaiter();

    for (uint32_t i = 1; i < GPUConst::levels_of_detail; ++i) {
        labels_.emplace_back(std::format("src view into layer {}", i));
        wgpu::TextureViewDescriptor src_desc{
            .label = labels_.back().c_str(),
            .format = wgpu::TextureFormat::R32Float,
            .dimension = wgpu::TextureViewDimension::e2D,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .usage = wgpu::TextureUsage::TextureBinding};

        labels_.emplace_back(std::format("src view into layer {}", i));
        wgpu::TextureViewDescriptor dest_desc{
            .label = labels_.back().c_str(),
            .format = wgpu::TextureFormat::R32Float,
            .dimension = wgpu::TextureViewDimension::e2D,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .usage = wgpu::TextureUsage::StorageBinding};

        auto callback = [&]() {
            texture_views_[i] = std::make_pair(
                shared_bindings_.getTexture(0).CreateView(&src_desc),
                shared_bindings_.getTexture(0).CreateView(&dest_desc));
        };

        awaiter.addCall(std::move(callback),
                        std::format("create view pair for lod {}", i));
    }

    return awaiter.executeAll();
}

std::optional<std::string> FillPyramidPass::initBindGroups() {
    spdlog::info(LOG_ID " initializing {} bind groups");

    auto awaiter = gpu_->getAwaiter();

    for (uint32_t lod = 1; lod < GPUConst::levels_of_detail; ++lod) {
        const std::array<wgpu::BindGroupEntry, 3> bindings{
            wgpu::BindGroupEntry{.binding = 0,
                                 .textureView = texture_views_.at(lod).first},
            wgpu::BindGroupEntry{.binding = 1, .sampler = sampler_},
            wgpu::BindGroupEntry{.binding = 2,
                                 .textureView = texture_views_.at(lod).second},
        };

        labels_.emplace_back(std::format("bind group for lod {}", lod));
        const wgpu::BindGroupDescriptor desc{
            .label = labels_.back().c_str(),
            .layout = bind_group_layout_,
            .entryCount = bindings.size(),
            .entries = bindings.data(),
        };

        auto callback
            = [&]() { bind_group_ = gpu_->getDevice().CreateBindGroup(&desc); };

        awaiter.addCall(std::move(callback), LOG_ID " creating bind group");
    }

    return awaiter.executeAll();
}

std::optional<std::string> FillPyramidPass::initComputePipeline() {
    auto mod = gpu_->loadShaderModule("create_pyramid.wgsl", "create pyramid");

    if (!mod) {
        return "creating shader module: " + mod.error();
    }

    wgpu::ComputePipelineDescriptor desc{
        .compute = {.module = mod.value(), .entryPoint = "main"},
    };

    auto awaiter = gpu_->getAwaiter();
    awaiter.addCall(
        [&]() {
            compute_pipeline_ = gpu_->getDevice().CreateComputePipeline(&desc);
        },
        LOG_ID " create compute pipeline");

    return awaiter.executeAll();
}

std::optional<std::string> FillPyramidPass::execute() {
    if (auto err = writeBaseLayer()) {
        return "writing base level (source frame): " + err.value();
    }

    if (auto err = writeNonBaseLayers()) {
        return "writing pyramid mips: " + err.value();
    }
    return std::nullopt;
}

std::optional<std::string> FillPyramidPass::writeBaseLayer() {
    wgpu::Queue queue = gpu_->getDevice().GetQueue();

    wgpu::TexelCopyTextureInfo copy_info{
        .texture = shared_bindings_.getTexture(0),
        .mipLevel = 0,
        .origin = {.x = 0, .y = 0},
        .aspect = wgpu::TextureAspect::All,
    };

    wgpu::TexelCopyBufferLayout copy_layout{
        .offset = 0,
        .bytesPerRow = GPUConst::pixel_size * (GPUConst::frame_width),
        .rowsPerImage = GPUConst::frame_height,
    };

    wgpu::Extent3D extent{.width = GPUConst::frame_width,
                          .height = GPUConst::frame_height,
                          .depthOrArrayLayers = 1};

    auto data = image_getter_();
    if (!data.has_value()) {
        return "end of data";
    }

    auto awaiter = gpu_->getAwaiter();

    auto callback = [&]() {
        queue.WriteTexture(&copy_info, data->data(), data->size_bytes(),
                           &copy_layout, &extent);
    };

    awaiter.addCall(std::move(callback), LOG_ID " write texture source");

    if (auto err = awaiter.executeAll()) {
        return "write source texture: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> FillPyramidPass::writeNonBaseLayers() {
    auto device = gpu_->getDevice();
    auto queue = device.GetQueue();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    const auto width = GPUConst::frame_width;
    const auto height = GPUConst::frame_height;
    const auto workgroup_size = 16;

    for (uint32_t lod = 1; lod < GPUConst::levels_of_detail; lod++) {
        labels_.emplace_back(std::format("fill lod {}", lod));

        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();

        pass.SetPipeline(compute_pipeline_);
        pass.SetLabel(labels_.back().c_str());
        pass.SetBindGroup(0, bind_groups_.at(lod));
        pass.DispatchWorkgroups(width / workgroup_size,
                                height / workgroup_size);

        pass.End();
    }

    auto commands = encoder.Finish();

    queue.Submit(1, &commands);

    auto awaiter = gpu_->getAwaiter();

    awaiter.addFuture(queue.OnSubmittedWorkDone(
                          wgpu::CallbackMode::WaitAnyOnly,
                          [](wgpu::QueueWorkDoneStatus status,
                             wgpu::StringView msg, decltype(this)) {
                              spdlog::info(
                                  LOG_ID
                                  " Submitted work done for writing "
                                  "pyramid levels; status: {}, msg: {}",
                                  int(status), std::string(msg));
                          },
                          this),
                      "Execute compute pass", false);
    return awaiter.executeAll().transform(
        [](const std::string& s) { return "exeuting: " + s; });
}
