#include "detect_corners.hpp"

#include <spdlog/spdlog.h>
#include <unistd.h>
#include <webgpu/webgpu_cpp.h>

#include "awaiter.hpp"
#include "common.hpp"

using namespace wslam;
using namespace std::chrono_literals;

namespace c = wslam::compute;

#define LOG_ID "[Pass Detect Corners]"

namespace {
const std::string kCornersBindings = "corners";
const std::string kPassParamsBinding = "pass_params";
};  // namespace

std::string PassDetectCorners::getId() const { return LOG_ID; }

std::optional<std::string> PassDetectCorners::initialize() {
    spdlog::info(getId() + " initializing");

    assert(gpu_->getDevice() != nullptr);
    assert(gpu_->getInstance() != nullptr);

    fillPassParams();
    fillConstants();

    if (auto err = initTextureViews()) {
        return "texture views: " + err.value();
    }

    if (auto err = initBindGroupLayouts()) {
        return "bind group layout: " + err.value();
    }

    if (auto err = initPerPassBindGroups()) {
        return "bind group: " + err.value();
    }

    if (auto err = initCommonBindGroup()) {
        return "common bind group: " + err.value();
    }

    if (auto err = initComputePipeline()) {
        return "compute pipeline: " + err.value();
    }

    return std::nullopt;
}

void PassDetectCorners::fillPassParams() {
    for (size_t i = 0; i < pass_params_.size(); i++) {
        pass_params_[i].lod = static_cast<uint32_t>(i);
    }
}

void PassDetectCorners::fillConstants() {
    compute_constants = {
        wgpu::ConstantEntry{
            .key = "SCALE_FACTOR",
            .value = GPUConst::lod_scale_factor,
        },
        {
            .key = "SRC_IMAGE_W",
            .value = GPUConst::frame_width,

        },
        {
            .key = "SRC_IMAGE_H",
            .value = GPUConst::frame_height,
        },
    };
}

std::optional<std::string> PassDetectCorners::initTextureViews() {
    spdlog::info(LOG_ID "initializing texture views into different LoDs");

    auto awaiter = gpu_->getAwaiter();

    for (size_t i = 0; i < texture_views_.size(); i++) {
        wgpu::TextureViewDescriptor desc{
            .label = std::format("view into LoD {}", i).c_str(),
            .format = wgpu::TextureFormat::R32Float,
            .dimension = wgpu::TextureViewDimension::e2D,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .usage = wgpu::TextureUsage::TextureBinding,
        };

        auto create_view = [&]() {
            texture_views_.at(i)
                = shared_bindings_.getTexture(i).CreateView(&desc);
        };

        awaiter.addCall(std::move(create_view),
                        std::format("create view into LoD {}", i));
    }

    return awaiter.executeAll().transform(
        [](const std::string& err) { return "awaiter: " + err; });
}

std::optional<std::string> PassDetectCorners::initBindGroupLayouts() {
    spdlog::info(LOG_ID " initializing bind group layouts");

    auto awaiter = gpu_->getAwaiter();

    if (auto err = initCornersBindGroupLayout(awaiter)) {
        return "corners: " + err.value();
    }

    if (auto err = initPassDataBindGroupLayout(awaiter)) {
        return "pass data: " + err.value();
    }

    return awaiter.executeAll().transform(
        [](const std::string& err) { return "awaiter: " + err; });
}

std::optional<std::string> PassDetectCorners::initCornersBindGroupLayout(
    compute::Awaiter& awaiter) {
    spdlog::debug(getId() + " initializing bind group layout for corners");

    const std::vector<wgpu::BindGroupLayoutEntry> bindings{
        {
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        },
    };

    wgpu::BindGroupLayoutDescriptor desc{
        .label = LOG_ID " Bind group layout",
        .entryCount = bindings.size(),
        .entries = bindings.data(),
    };

    awaiter.addCall(
        [&]() {
            bind_group_layouts_.corners
                = gpu_->getDevice().CreateBindGroupLayout(&desc);
        },
        LOG_ID " create bind group layout for corners");

    return std::nullopt;
}

std::optional<std::string> PassDetectCorners::initPassDataBindGroupLayout(
    compute::Awaiter& awaiter) {
    spdlog::debug(getId()
                  + " initializing bind group layout for pass-specific data");

    const std::vector<wgpu::BindGroupLayoutEntry> bindings{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .texture = {.sampleType = wgpu::TextureSampleType::Float,
                        .viewDimension = wgpu::TextureViewDimension::e2D,}

        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Uniform },
        },
    };

    wgpu::BindGroupLayoutDescriptor desc{
        .label = LOG_ID " per-pass bind group layout ",
        .entryCount = bindings.size(),
        .entries = bindings.data(),
    };

    awaiter.addCall(
        [&]() {
            bind_group_layouts_.pass_data
                = gpu_->getDevice().CreateBindGroupLayout(&desc);
        },
        LOG_ID " create bind group layout for pass-specific-data");

    return std::nullopt;
}

std::optional<std::string> PassDetectCorners::initCommonBindGroup() {
    spdlog::info(LOG_ID " initializing bind group common between passes");

    wgpu::BindGroupEntry entry{
        .binding = 0,
        .size = kConrnersArraySize,
    };

    auto buf_binding = gpu_->assignBuffersAndOffsets(
        {{kCornersBindings,
          compute::GPU::BgBinding{.buf_type = compute::BufferType::StorageB,
                                  .bg_entry = entry}}});

    if (!buf_binding) {
        return "assingiing buffer bindings: " + buf_binding.error();
    }

    buf_bindings_.merge(buf_binding.value());

    assert(buf_binding->empty() && "Values moved out successfully");

    wgpu::BindGroupDescriptor desc{
        .label = "Common bind group for " LOG_ID,
        .layout = bind_group_layouts_.corners,
        .entryCount = 1,
        .entries = &entry,
    };
    auto create_bg = [&]() {
        common_bind_group_ = gpu_->getDevice().CreateBindGroup(&desc);
    };

    return gpu_->getAwaiter()
        .addCall(std::move(create_bg), "create bind group for corners")
        .executeAll()
        .transform([](const auto& err) { return "awaiter: " + err; });
}

std::optional<std::string> PassDetectCorners::initPerPassBindGroups() {
    spdlog::info(getId() + " initializing pass-specific bind groups");

    auto awaiter = gpu_->getAwaiter();

    for (const auto& [key, bind] : buf_bindings_) {
        spdlog::warn(LOG_ID " existing buffer binding: {}", bind);
    }

    for (size_t i = 0; i < per_pass_bind_groups_.size(); i++) {
        std::array<wgpu::BindGroupEntry, 2> entries{
            wgpu::BindGroupEntry{
                .binding = 0,
                .textureView = texture_views_.at(i),
            },
            wgpu::BindGroupEntry{.binding = 1, .size = sizeof(PassParams)},
        };

        const std::string label = std::format("bind group for LoD {}", i);
        const std::string binding_label
            = kPassParamsBinding + ":lod:" + std::to_string(i);

        const wgpu::BindGroupDescriptor desc{
            .label = {label},
            .layout = bind_group_layouts_.pass_data,
            .entryCount = entries.size(),
            .entries = entries.data(),
        };

        auto bindings = gpu_->assignBuffersAndOffsets({{
            binding_label,
            compute::GPU::BgBinding{
                .buf_type = compute::BufferType::Uniform,
                .bg_entry = entries.at(1),
            },
        }});

        if (!bindings) {
            return std::format("assigning buffers and offsets for LoD {}: {}",
                               i, bindings.error());
        }

        assert(bindings->size() == 1);

        spdlog::debug(LOG_ID " inserting binding key: '{}'",
                      bindings->begin()->first);

        buf_bindings_.merge(bindings.value());

        if (!bindings->empty()) {
            for (const auto& [key, bind] : bindings.value()) {
                spdlog::error(LOG_ID
                              " duplicate binding after trying to move new "
                              "buffer bindings. Binding: {}",
                              std::format("{}", bind));
            }
            throw std::logic_error(
                "duplicate bindings (double initialization?)");
        }

        auto create_bg = [&]() {
            per_pass_bind_groups_.at(i)
                = gpu_->getDevice().CreateBindGroup(&desc);
        };

        awaiter.addCall(std::move(create_bg),
                        "create bind group for LoD " + std::to_string(i));
    }

    return awaiter.executeAll();
}

std::optional<std::string> PassDetectCorners::initComputePipeline() {
    spdlog::info(getId() + " initializing compute pipeline");

    wgpu::ShaderModule module;

    auto mod_err
        = gpu_->loadShaderModule("detect_corners.wgsl", "detect corners");

    if (!mod_err) {
        return "loading shader module: " + mod_err.error();
    }

    module = std::move(mod_err.value());

    auto awaiter = gpu_->getAwaiter();

    wgpu::PipelineLayoutDescriptor layout_desc{
        .label = LOG_ID " pipeline layout",
        .bindGroupLayoutCount
        = wslam::PassDetectCorners::BindGroupLayouts::size(),
        .bindGroupLayouts = bind_group_layouts_.data(),
    };

    auto create_layout = [&]() {
        compute_pipeline_layout_
            = gpu_->getDevice().CreatePipelineLayout(&layout_desc);
    };

    awaiter.addCall(std::move(create_layout), "create pipeline layout");

    wgpu::ComputePipelineDescriptor compute_desc{
        .label = LOG_ID " compute pipeline",
        .layout = compute_pipeline_layout_,
        .compute = {.module = module,
                    .entryPoint = "main",
                    .constantCount = compute_constants.size(),
                    .constants = compute_constants.data()},
    };

    auto create_pipeline = [&]() {
        compute_pipeline_
            = gpu_->getDevice().CreateComputePipeline(&compute_desc);
    };
    awaiter.addCall(std::move(create_pipeline), "create compute pipeline");

    return awaiter.executeAll().transform(
        [](const auto& err) { return "awaiter: " + err; });
}

std::optional<std::string> PassDetectCorners::execute() {
    const wgpu::Device device = gpu_->getDevice();
    const wgpu::Queue queue = device.GetQueue();

    std::array<wgpu::CommandBuffer, 2> command_buffers;

    // WRITE PARAMS FOR ALL PASSES
    const wgpu::CommandEncoderDescriptor params_desc{
        .label = LOG_ID " write params for passes",
    };
    const auto write_params_encoder = device.CreateCommandEncoder(&params_desc);
    if (auto err = writeGPUPassParams(write_params_encoder)) {
        return "writing gpu pass parameters: " + err.value();
    }

    command_buffers.at(0) = write_params_encoder.Finish();

    // LAUNCH MAIN PASSES
    const auto compute_encoder = device.CreateCommandEncoder();

    for (size_t i = 0; i < GPUConst::levels_of_detail; i++) {
        const wgpu::ComputePassDescriptor pass_desc{
            .label = std::format("detect corners in LoD {}", i).c_str(),
        };
        const auto pass = compute_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(compute_pipeline_);
        pass.SetBindGroup(0, common_bind_group_);
        pass.SetBindGroup(1, per_pass_bind_groups_.at(i));

        pass.End();
    }

    command_buffers.at(1) = compute_encoder.Finish();

    queue.Submit(command_buffers.size(), command_buffers.data());

    std::string status_error;

    auto on_work_done = [&]() -> wgpu::Future {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView msg,
               std::string* error) {
                spdlog::info(LOG_ID " execution finished. status:{}, msg:{}",
                             static_cast<int>(status),
                             static_cast<std::string>(msg));
                if (status != wgpu::QueueWorkDoneStatus::Success) {
                    *error = msg;
                }
            },
            &status_error);
    };

    auto awaiter = gpu_->getAwaiter();
    awaiter.addCall(std::move(on_work_done), "execute gpu compute pass");

    if (auto err = awaiter.executeAll()) {
        return "awaiter: " + err.value();
    }

    if (status_error.length() > 0) {
        return "unsuccessful exeuction: " + status_error;
    }

    return std::nullopt;
}

std::optional<std::string> PassDetectCorners::writeGPUPassParams(
    const wgpu::CommandEncoder& encoder) {
    const auto data = std::as_bytes(std::span(pass_params_));
    auto err = gpu_->fillNonInputBuffer(encoder, data,
                                        buf_bindings_.at(kPassParamsBinding));

    return err;
}
