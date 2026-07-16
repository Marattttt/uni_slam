#include "cull_corners.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <string>

#include "common.hpp"
#include "compute/gpu.hpp"

using namespace wslam;

#define LOG_ID "[Cull Corners pass]"

namespace {
constinit std::string kVerticalBindingLabel = "vertical";
constexpr std::string_view kShaderModulePath = "cull_corners.wgsl";

// Sholud be better for memory access, not tested as of writing
constexpr std::array<uint32_t, 3> kWgSize = {64, 1, 1};
constexpr uint32_t kRegionSize = 3;
std::array<compute::GPU::ShaderOverride, 2> GetShaderOverrides() {
    return {
        compute::GPU::ShaderOverride{
            "LOD_COUNT",
            std::format("{}u", GPUConst::levels_of_detail),
        },

        compute::GPU::ShaderOverride{
            "CORNER_BLOCK_SZ",
            std::format("{}u", GPUConst::frame_height * GPUConst::frame_width)},
    };
}
}  // namespace

std::string CullCornersPass::getId() const { return LOG_ID; }

std::optional<std::string> CullCornersPass::initialize() {
    spdlog::info(LOG_ID " Initializing");

    std::optional<std::string> err;

    initGpuDataEntries();
    initComputeConstants();

    err = initBindingLayout();
    if (err) {
        return ":" + err.value();
    }
    err = initBindGroups();
    if (err) {
        return ":" + err.value();
    }
    err = initComputePipeline();
    if (err) {
        return ":" + err.value();
    }

    return std::nullopt;
}

void CullCornersPass::initGpuDataEntries() {
    assert(gpu_data_.size() == 0);

    const PassGpuData empty{.bg = {},
                            .bg_layout = bind_group_layout_,
                            .pipeline = {},
                            .pipeline_layout = compute_pipeline_layout_};

    gpu_data_.insert_or_assign(Workflow::Horizontal, empty);
    gpu_data_.insert_or_assign(Workflow::Vertical, empty);
}

std::optional<std::string> CullCornersPass::initBindingLayout() {
    spdlog::debug(LOG_ID " Initializing bindings layouts");

    const std::array<wgpu::BindGroupLayoutEntry, 2> entries{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        },
    };

    const wgpu::BindGroupLayoutDescriptor desc{
        .label = LOG_ID " Bind group layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };

    auto create = [&]() {
        bind_group_layout_ = gpu_->getDevice().CreateBindGroupLayout(&desc);
    };

    return gpu_->getAwaiter()
        .runChecked(create, "Create bind group layout")
        .executeAll();
}

// TODO: this needs rewriting
std::optional<std::string> CullCornersPass::initBindGroups() {
    spdlog::info(LOG_ID " Initializing bind groups");

    auto inputBinding
        = shared_bindings_.getStorage().getPtr<compute::BufferBinding>(
            kInputCornersLabel);

    if (!inputBinding) {
        return "could not get input corner binding";
    }

    using BT = compute::BufferType;

    const auto* unculled = inputBinding.value();

    const auto unculled_buf_type = unculled->getBuffertype();
    assert(unculled_buf_type == BT::StorageA
           || unculled_buf_type == BT::StorageB);

    // Opposite storage buffer for the first pass
    const auto horizontal_buftype
        = unculled_buf_type == BT::StorageA ? BT::StorageB : BT::StorageA;
    const auto vertical_buftype = unculled_buf_type;

    const auto binding_size = unculled->getSize();

    std::array<wgpu::BindGroupEntry, 2> vertical_entries{
        wgpu::BindGroupEntry{.binding = 0, .size = binding_size},
        wgpu::BindGroupEntry{.binding = 1, .size = binding_size},
    };

    auto bindings = gpu_->assignBuffersAndOffsets({
        {
            kVerticalBindingLabel,
            compute::GPU::BgBinding{.buf_type = horizontal_buftype,
                                    .bg_entry = vertical_entries.at(0)},
        },
        {
            kCulledCornersOutputLabel,
            compute::GPU::BgBinding{
                .buf_type = vertical_buftype,
                .bg_entry = vertical_entries.at(1),
            },
        },
    });

    if (!bindings) {
        return "reserving buffers: " + bindings.error();
    }

    for (auto& [key, val] : bindings.value()) {
        shared_bindings_.getStorage().set(key, std::move(val));
    }

    std::array<wgpu::BindGroupEntry, 2> horizontal_entries{
        wgpu::BindGroupEntry{.binding = 0,
                             .buffer = gpu_->getBuffer(unculled_buf_type),
                             .offset = unculled->getOffset(),
                             .size = unculled->getSize()},
        vertical_entries.at(0),
    };
    horizontal_entries.at(1).binding = 1;

    const wgpu::BindGroupDescriptor horizontal_desc{
        .label = LOG_ID " Horizontal culling bind group",
        .layout = bind_group_layout_,
        .entryCount = horizontal_entries.size(),
        .entries = horizontal_entries.data(),
    };

    auto create_horizontal = [&] {
        const auto bg = gpu_->getDevice().CreateBindGroup(&horizontal_desc);
        gpu_data_.at(Workflow::Horizontal).bg = bg;
    };

    if (auto err = gpu_->getAwaiter()
                       .runChecked(create_horizontal,
                                   "Create bind group for horizontal culling")
                       .executeAll()) {
        return "horizontal: " + err.value();
    }

    const wgpu::BindGroupDescriptor vertical_descriptor{
        .label = LOG_ID " Vertical culling bind group",
        .layout = bind_group_layout_,
        .entryCount = vertical_entries.size(),
        .entries = vertical_entries.data(),
    };

    auto create_vertical = [&] {
        const auto bg = gpu_->getDevice().CreateBindGroup(&vertical_descriptor);
        gpu_data_.at(Workflow::Vertical).bg = bg;
    };

    if (auto err = gpu_->getAwaiter()
                       .runChecked(create_vertical,
                                   "Create bind group for vertical cullinng")
                       .executeAll()) {
        return "vertical: " + err.value();
    }

    return std::nullopt;
}

void CullCornersPass::initComputeConstants() {
    compute_constants_ = {
        wgpu::ConstantEntry{.key = "SRC_IMAGE_W",
                            .value = GPUConst::frame_width},
        {.key = "SRC_IMAGE_H", .value = GPUConst::frame_height},
        {.key = "WG_SIZE_X", .value = kWgSize.at(0)},
        {.key = "WG_SIZE_Y", .value = kWgSize.at(1)},
    };
}

std::optional<std::string> CullCornersPass::initComputePipeline() {
    spdlog::info(LOG_ID " Initializing compute pipeline");

    const wgpu::PipelineLayoutDescriptor layout_desc{
        .label = LOG_ID " pipeline layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bind_group_layout_,
    };

    auto create_layout = [&] {
        compute_pipeline_layout_
            = gpu_->getDevice().CreatePipelineLayout(&layout_desc);
    };

    if (auto err = gpu_->getAwaiter()
                       .runChecked(create_layout, "Create pipeline layout")
                       .executeAll()) {
        return "pipeline layout: " + err.value();
    }

    auto code_err = gpu_->loadShaderModule(kShaderModulePath, "cull corners",
                                           GetShaderOverrides());
    if (!code_err) {
        return "loading shader: " + code_err.error();
    }
    const auto code = std::move(code_err.value());

    const wgpu::ComputePipelineDescriptor horizontal_desc {
        .label = LOG_ID " horizontal compute pipeline",
            .layout = compute_pipeline_layout_,
            .compute = {
                .module = code,
                .entryPoint = "main_horizontal",
                .constantCount = compute_constants_.size(),
                .constants = compute_constants_.data(),
            },
    };

    const wgpu::ComputePipelineDescriptor vertical_desc {
        .label = LOG_ID " vertical compute pipeline",
        .layout = compute_pipeline_layout_,
        .compute = {
            .module = code,
            .entryPoint = "main_vertical",
            .constantCount = compute_constants_.size(),
            .constants = compute_constants_.data(),
        },
    };

    const auto create_pipeline
        = [&](const wgpu::ComputePipelineDescriptor& desc, Workflow workflow) {
              return [=, this] {
                  const auto pipeline
                      = gpu_->getDevice().CreateComputePipeline(&desc);
                  gpu_data_.at(workflow).pipeline = pipeline;
              };
          };

    auto awaiter = gpu_->getAwaiter();

    awaiter.runChecked(create_pipeline(horizontal_desc, Workflow::Horizontal),
                       "Horizontal pass pipeline");
    awaiter.runChecked(create_pipeline(vertical_desc, Workflow::Vertical),
                       "Vertical pass pipeline");

    if (auto err = awaiter.executeAll()) {
        return "crating compute pipelines: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> CullCornersPass::prepareExecute(
    const wgpu::CommandEncoder& encoder) {
    spdlog::info(LOG_ID " Executing");

    writeWorkflowPass(Workflow::Horizontal, encoder);
    writeWorkflowPass(Workflow::Vertical, encoder);

    return std::nullopt;
}

void CullCornersPass::writeWorkflowPass(
    Workflow workflow, const wgpu::CommandEncoder& encoder) const {
    const auto pass = encoder.BeginComputePass();

    const std::string label = std::invoke([workflow] {
        switch (workflow) {
            case Workflow::Vertical:
                return "vertical";
            case Workflow::Horizontal:
                return "horizontal";
        }
        std::unreachable();
    });

    const auto& data = gpu_data_.at(workflow);

    pass.SetLabel({label});
    pass.SetPipeline(data.pipeline);
    pass.SetBindGroup(0, data.bg);

    const auto dispatch_full = getDispatchSize();
    const auto dispatch = workflow == Workflow::Horizontal
                              ? dispatch_full.at(0)
                              : dispatch_full.at(1);

    pass.DispatchWorkgroups(dispatch.at(0), dispatch.at(1), dispatch.at(2));

    pass.End();
}

std::array<std::array<uint32_t, 3>, 2> CullCornersPass::getDispatchSize() {
    const uint32_t horx = CeilDiv(GPUConst::frame_width, kRegionSize);
    const uint32_t hory = GPUConst::frame_height;
    const uint32_t horz = GPUConst::levels_of_detail;

    constexpr std::array<uint32_t, 3> horizontal{
        CeilDiv(horx, kWgSize[0]),
        CeilDiv(hory, kWgSize[1]),
        CeilDiv(horz, kWgSize[2]),
    };

    const uint32_t vert_x = GPUConst::frame_width;
    const uint32_t vert_y = CeilDiv(GPUConst::frame_height, kRegionSize);
    const uint32_t vert_z = GPUConst::levels_of_detail;
    constexpr std::array<uint32_t, 3> vertical{
        CeilDiv(vert_x, kWgSize[0]),
        CeilDiv(vert_y, kWgSize[1]),
        CeilDiv(vert_z, kWgSize[2]),
    };

    return {horizontal, vertical};
}
