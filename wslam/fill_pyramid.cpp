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
        return "initSampler: " + err.value();
    }
    if (auto err = initBindGroupLayout()) {
        return "initBindGroupLayout: " + err.value();
    }
    if (auto err = initBindGroup()) {
        return "initBindGroup: " + err.value();
    }
    if (auto err = initComputePipeline()) {
        return "initComputePipeline: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> FillPyramidPass::initBindGroupLayout() {
    spdlog::info(LOG_ID " initializing bind group layout");

    std::array<wgpu::BindGroupLayoutEntry, 1> layout_entries{};

    // Sampler
    layout_entries[0].binding = 0;
    layout_entries[0].visibility = wgpu::ShaderStage::Compute;
    layout_entries[0].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bgl_desc{};
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

std::optional<std::string> FillPyramidPass::initBindGroup() {
    spdlog::info(LOG_ID " initializing bind group");

    const std::array<wgpu::BindGroupEntry, 1> bindings{
        wgpu::BindGroupEntry{.binding = 0, .sampler = sampler_},
    };

    const wgpu::BindGroupDescriptor desc{
        .label = LOG_ID " bind group",
        .layout = bind_group_layout_,
        .entryCount = bindings.size(),
        .entries = bindings.data(),
    };

    auto callback
        = [&]() { bind_group_ = gpu_->getDevice().CreateBindGroup(&desc); };

    auto awaiter = gpu_->getAwaiter();
    awaiter.addCall(std::move(callback), LOG_ID " creating bind group");

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
    if (auto err = writeBaseMip()) {
        return "writing base mip level (source frame): " + err.value();
    }

    if (auto err = writePyramidMips()) {
        return "writing pyramid mips: " + err.value();
    }
    return std::nullopt;
}

std::optional<std::string> FillPyramidPass::writeBaseMip() {
    wgpu::Queue queue = gpu_->getDevice().GetQueue();

    wgpu::TexelCopyTextureInfo copy_info{
        .texture = shared_bindings_.frame_,
        .mipLevel = 0,
        .origin = {.x = 0, .y = 0},
        .aspect = wgpu::TextureAspect::All,
    };

    const uint32_t u8max = std::numeric_limits<uint8_t>::max();

    wgpu::TexelCopyBufferLayout copy_layout{
        .offset = 0,
        .bytesPerRow = (GPUConst::frame_width * 1 + u8max) & ~u8max,
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

std::optional<std::string> FillPyramidPass::writePyramidMips() {
    auto device = gpu_->getDevice();
    auto queue = device.GetQueue();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    std::vector<wgpu::TextureView> views;
    std::vector<std::string> view_labels;

    for (uint32_t mip = 1; mip < GPUConst::pyr_levels; mip++) {
        const uint32_t dst_w = std::max(1U, GPUConst::frame_width >> mip);
        const uint32_t dst_h = std::max(1U, GPUConst::frame_height >> mip);

        view_labels.emplace_back(
            std::format("read-only view into mip #{}", mip - 1));

        // Source: previous mip level (read via sampler)
        wgpu::TextureViewDescriptor src_view_desc{
            .label = view_labels.back().c_str(),
            .format = wgpu::TextureFormat::RGBA8Unorm,
            .dimension = wgpu::TextureViewDimension::e2D,
            .baseMipLevel = mip - 1,
            .mipLevelCount = 1,
        };
        wgpu::TextureView src_view
            = shared_bindings_.frame_.CreateView(&src_view_desc);

        view_labels.emplace_back(
            std::format("write-only view into mip #{}", mip));

        // Destination: current mip level (storage write)
        wgpu::TextureViewDescriptor dst_view_desc{
            .label = view_labels.back().c_str(),
            .format = wgpu::TextureFormat::RGBA8Unorm,
            .dimension = wgpu::TextureViewDimension::e2D,
            .baseMipLevel = mip,
            .mipLevelCount = 1,
        };
        wgpu::TextureView dst_view
            = shared_bindings_.frame_.CreateView(&dst_view_desc);

        views.emplace_back(std::move(src_view));
        views.emplace_back(std::move(dst_view));

        // BindGroup for this mip
        std::array<wgpu::BindGroupEntry, 3> entries{};
        entries[0] = {.binding = 0, .textureView = views.at(views.size() - 2)};
        entries[1] = {.binding = 1, .sampler = sampler_};
        entries[2] = {.binding = 2, .textureView = views.at(views.size() - 1)};

        wgpu::BindGroupDescriptor bg_desc{};
        bg_desc.layout = compute_pipeline_.GetBindGroupLayout(0);
        bg_desc.entryCount = entries.size();
        bg_desc.entries = entries.data();
        wgpu::BindGroup bind_group = device.CreateBindGroup(&bg_desc);

        // Dispatch
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(compute_pipeline_);
        pass.SetBindGroup(1, bind_group);
        pass.SetBindGroup(0, shared_bindings_.group);
        pass.DispatchWorkgroups((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
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
