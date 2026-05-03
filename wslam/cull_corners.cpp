#include "cull_corners.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include "awaiter.hpp"
#include "common.hpp"
#include "gpu.hpp"

using namespace wslam;

#define LOG_ID "[Cull Corners pass]"

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
        .addCall(std::move(create), "Create bind group layout")
        .executeAll();
}

// TODO: this needs rewriting
std::optional<std::string> CullCornersPass::initBindGroups() {
    spdlog::info(LOG_ID " Initializing bind groups");

    auto inputBinding
        = shared_bindings_.getStorage().take<compute::BufferBinding>(
            kInputCornersLabel);

    if (!inputBinding) {
        return "could not get input corner binding";
    }

    shared_bindings_.getStorage().set(kCulledCornersOutputLabel,
                                      std::move(inputBinding.value()));

    using BT = compute::BufferType;

    const auto* unculled
        = shared_bindings_.getStorage()
              .getPtr<compute::BufferBinding>(kCulledCornersOutputLabel)
              .value();

    const auto unculled_buf_type = unculled->getBuffertype();
    assert(unculled_buf_type == BT::StorageA
           || unculled_buf_type == BT::StorageB);

    // Opposite storage buffer for the first pass, and same binding used for the
    // second pass
    const auto horizontal_buftype
        = unculled_buf_type == BT::StorageA ? BT::StorageB : BT::StorageA;
    const auto binding_size = unculled->getSize();

    wgpu::BindGroupEntry temp_bg_entry{
        .binding = 0,
        .size = binding_size,
    };

    auto temp_binding = gpu_->assignBuffersAndOffsets({{
        std::string{kVerticalBindingLabel},
        compute::GPU::BgBinding{.buf_type = horizontal_buftype,
                                .bg_entry = temp_bg_entry},
    }});

    if (!temp_binding) {
        return "assigning buffer region for vertical buffer: "
               + temp_binding.error();
    }

    vertical_binding_ = std::move(
        temp_binding.value().at(std::string(kVerticalBindingLabel)));

    // HORIZONTAL PASS
    spdlog::debug(LOG_ID " Initializing bind group for horizontal pass");
    const std::array<wgpu::BindGroupEntry, 2> horizontal_entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = gpu_->getBuffer(unculled->getBuffertype()),
            .offset = unculled->getOffset(),
            .size = binding_size},
        wgpu::BindGroupEntry{
            .binding = 1,
            .buffer = temp_bg_entry.buffer,
            .offset = temp_bg_entry.offset,
            .size = binding_size,
        },
    };

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
                       .addCall(std::move(create_horizontal),
                                "Create bind group for horizontal culling")
                       .executeAll()) {
        return "horizontal: " + err.value();
    }

    // VERTICAL PASS
    spdlog::debug(LOG_ID " Initializing bind group for vertical pass");

    std::array<wgpu::BindGroupEntry, 2> vertical_entries{
        horizontal_entries.at(1),
        horizontal_entries.at(0),
    };
    vertical_entries.at(0).binding = 0;
    vertical_entries.at(1).binding = 1;

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
                       .addCall(std::move(create_vertical),
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

    if (auto err
        = gpu_->getAwaiter()
              .addCall(std::move(create_layout), "Create pipeline layout")
              .executeAll()) {
        return "pipeline layout: " + err.value();
    }

    auto code_err = gpu_->loadShaderModule(kShaderModulePath, "cull corners");
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

    awaiter.addCall(create_pipeline(horizontal_desc, Workflow::Horizontal),
                    "Horizontal pass pipeline");
    awaiter.addCall(create_pipeline(vertical_desc, Workflow::Vertical),
                    "Vertical pass pipeline");

    if (auto err = awaiter.executeAll()) {
        return "crating compute pipelines: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> CullCornersPass::execute() {
    spdlog::info(LOG_ID " Executing");

    const auto device = gpu_->getDevice();
    const auto queue = device.GetQueue();
    const auto encoder = device.CreateCommandEncoder();

    writeWorkflowPass(Workflow::Horizontal, encoder);
    writeWorkflowPass(Workflow::Vertical, encoder);

    const auto commands = encoder.Finish();

    queue.Submit(1, &commands);

    std::string errormsg;

    auto wait_submission = [&] -> wgpu::Future {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView sv,
               std::string* err) {
                spdlog::info(LOG_ID " gpu work done. status:{} msg:'{}'",
                             static_cast<int>(status),
                             static_cast<std::string>(sv));
                if (status != wgpu::QueueWorkDoneStatus::Success) {
                    *err = std::string(sv);
                }
            },
            &errormsg);
    };

    const auto err
        = gpu_->getAwaiter()
              .addCall(std::move(wait_submission), "Wait GPU work", false)
              .executeAll();

    if (err) {
        spdlog::error(LOG_ID "Error executing gpu pass: {}", errormsg);
        return "executing: " + err.value();
    }

    const auto* binding
        = shared_bindings_.getStorage()
              .getPtr<compute::BufferBinding>(kCulledCornersOutputLabel)
              .value();
    const auto data = gpu_->readBuffer(*binding);

    if (!data) {
        return "reading culled data: " + data.error();
    }

    const auto nonzero = std::ranges::count_if(
        data.value(), [](const auto x) { return static_cast<int>(x) > 0; });

    spdlog::info(LOG_ID " Reading culled data. size:{} non-zero entries:{}",
                 data.value().size(), nonzero);

    return std::nullopt;
}

namespace {
std::array<uint32_t, 3> getDispatchSize(uint32_t wg_x, uint32_t wg_y) {
    const auto x = std::ceil(GPUConst::frame_width / wg_x);
    const auto y = std::ceil(GPUConst::frame_width / wg_y);
    const auto z = GPUConst::levels_of_detail;

    return {static_cast<uint32_t>(x), static_cast<uint32_t>(y), z};
}
};  // namespace

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

    auto dispatch = getDispatchSize(kWgSize.at(0), kWgSize.at(1));
    pass.DispatchWorkgroups(dispatch.at(0), dispatch.at(1), dispatch.at(2));

    pass.End();
}
