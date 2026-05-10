#include "generate_features.hpp"

#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <expected>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "common.hpp"
#include "gpu.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[Generate Descriptors pass]"
std::string GenerateFeaturesPass::getId() const { return LOG_ID; }

std::optional<std::string> GenerateFeaturesPass::initialize() {
    spdlog::info(LOG_ID " Initializing");

    const auto label
        = [](auto&& expected,
             const std::string& label) -> std::optional<std::string> {
        if (expected.has_value()) {
            return std::nullopt;
        }
        return label + ": "
               + std::forward<decltype(expected)>(expected).error();
    };

    if (auto err = label(initSampler(), "sampler")) {
        return err;
    }
    if (auto err = label(initBindGroupLayouts(), "bind group layout")) {
        return err;
    }
    if (auto err = label(initBindGroups(), "bind groups")) {
        return err;
    }
    if (auto err = label(initComputePipeline(), "compute pipeline")) {
        return err;
    }

    throw std::logic_error(LOG_ID "::initialize is not implemented!");

    return std::nullopt;
}

std::expected<void, std::string> GenerateFeaturesPass::initSampler() {
    spdlog::info(LOG_ID " Initializing sampler");

    const wgpu::SamplerDescriptor desc{
        .addressModeU = wgpu::AddressMode::ClampToEdge,
        .addressModeV = wgpu::AddressMode::ClampToEdge,
        .magFilter = wgpu::FilterMode::Linear,
        .minFilter = wgpu::FilterMode::Linear,
        .mipmapFilter = wgpu::MipmapFilterMode::Nearest,
    };

    auto create = [&] { sampler_ = gpu_->getDevice().CreateSampler(&desc); };

    return gpu_->getAwaiter()
        .addCall(std::move(create), "Create sampler")
        .execute();
}

std::expected<void, std::string> GenerateFeaturesPass::initBindGroupLayouts() {
    spdlog::info(LOG_ID " Initializing bind group layout");

    const auto common_res = initCommonBindgroupLayout().transform_error(
        [](const auto& err) { return "common: " + err; });

    if (!common_res) {
        return common_res;
    }

    const auto perpass_res = initPerPassBindgroupLayout().transform_error(
        [](const auto& err) { return "per pass layout: " + err; });

    if (!perpass_res) {
        return perpass_res;
    }

    return {};
}

std::expected<void, std::string>
GenerateFeaturesPass::initCommonBindgroupLayout() {
    spdlog::debug(LOG_ID " Initializing common bind group");

    const std::array<wgpu::BindGroupLayoutEntry, 3> entries{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Uniform},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        },
    };

    const wgpu::BindGroupLayoutDescriptor desc{
        .label = LOG_ID " common binding group",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };

    auto create = [&] {
        bind_group_layouts_.at(0)
            = gpu_->getDevice().CreateBindGroupLayout(&desc);
    };

    return gpu_->getAwaiter()
        .addCall(std::move(create), "Create layout for common bind group")
        .execute();
}

std::expected<void, std::string>
GenerateFeaturesPass::initPerPassBindgroupLayout() {
    spdlog::debug(LOG_ID " Initializing per pass bind group");

    const std::array<wgpu::BindGroupLayoutEntry, 2> entries{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .texture = {.sampleType = wgpu::TextureSampleType::Float,
                        .viewDimension = wgpu::TextureViewDimension::e2D},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .sampler = {.type = wgpu::SamplerBindingType::Filtering},
        },
    };

    const wgpu::BindGroupLayoutDescriptor desc{
        .label = LOG_ID " per pass binding group",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };

    auto create = [&] {
        bind_group_layouts_.at(1)
            = gpu_->getDevice().CreateBindGroupLayout(&desc);
    };

    return gpu_->getAwaiter()
        .addCall(std::move(create), "Create layout for per pass bind group")
        .execute();
}

std::expected<void, std::string> GenerateFeaturesPass::initBindGroups() {
    spdlog::info(LOG_ID " Initiializing bind groups");

    const auto common_res = initCommonBindgroup().transform_error(
        [](const auto& err) { return "common bindgroup: " + err; });
    if (!common_res) {
        return common_res;
    }

    const auto perpass_res = initPerPassBindgroups().transform_error(
        [](auto&& err) { return "per pass bind groups: " + err; });

    if (!perpass_res) {
        return perpass_res;
    }

    return {};
}

[[nodiscard]] std::expected<void, std::string>
GenerateFeaturesPass::initCommonBindgroup() {
    spdlog::debug(LOG_ID " Initializing common bind group");

    const auto corners_binding
        = shared_.getStorage()
              .take<compute::BufferBinding>(kCornersLabel)
              .value();

    using BT = compute::BufferType;

    const auto features_storage_type
        = corners_binding.getBuffertype() == BT::StorageA ? BT::StorageB
                                                          : BT::StorageA;

    assert(kCornersBindingSize == corners_binding.getSize());

    std::array<wgpu::BindGroupEntry, 3> entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = gpu_->getBuffer(corners_binding.getBuffertype()),
            .offset = corners_binding.getOffset(),
            .size = kCornersBindingSize},
        wgpu::BindGroupEntry{.binding = 1, .size = kBriefTestsBindingSize},
        wgpu::BindGroupEntry{.binding = 2, .size = kFeatureArrayBindingSize},
    };

    static constexpr auto kBriefTestsBindingLabel = "BRIEF tests";

    auto bindings
        = gpu_->assignBuffersAndOffsets(compute::GPU::BindGroupBindings{
            {
                kBriefTestsBindingLabel,
                compute::GPU::BgBinding{
                    .buf_type = compute::BufferType::Uniform,
                    .bg_entry = entries.at(1)},
            },
            {
                kFeaturesLabel,
                compute::GPU::BgBinding{.buf_type = features_storage_type,
                                        .bg_entry = entries.at(2)},
            },
        });

    if (!bindings) {
        return std::unexpected("requesting buffer regions: "
                               + std::move(bindings).error());
    }

    shared_.getStorage().set(kFeaturesLabel,
                             std::move(bindings.value().at(kFeaturesLabel)));

    brief_tests_binding_
        = std::move(bindings.value().at(kBriefTestsBindingLabel));

    const wgpu::BindGroupDescriptor desc{
        .label = LOG_ID " common bind group",
        .layout = bind_group_layouts_.at(0),
        .entryCount = entries.size(),
        .entries = entries.data(),
    };

    auto create = [&] {
        const auto bg = gpu_->getDevice().CreateBindGroup(&desc);
        for (auto& bgarray : bind_groups_) {
            bgarray.at(0) = bg;
        }
    };

    return gpu_->getAwaiter()
        .addCall(std::move(create), "create bind group")
        .execute()
        .transform_error(
            [](const auto& err) { return "creating bind groups: " + err; });
}

[[nodiscard]] std::expected<void, std::string>
GenerateFeaturesPass::initPerPassBindgroups() {
    spdlog::debug(LOG_ID " Initializing per pass bind groups");

    auto awaiter = gpu_->getAwaiter();

    for (size_t pass = 0; pass < kPassCount; pass++) {
        const auto texture
            = shared_.getStorage()
                  .get<wgpu::Texture>(ResourceIdentifier::GetFrameName(
                      {0, LOD{static_cast<uint8_t>(pass)}}))
                  .value();

        const std::string view_label
            = std::format(LOG_ID " view into lod {}", pass);

        const wgpu::TextureViewDescriptor view_desc{
            .label = view_label.c_str(),
            .format = wgpu::TextureFormat::R32Float,
            .dimension = wgpu::TextureViewDimension::e2D,
            .usage = wgpu::TextureUsage::TextureBinding};

        const auto view = texture.CreateView(&view_desc);

        const std::array<wgpu::BindGroupEntry, 2> entries{
            wgpu::BindGroupEntry{.binding = 0, .textureView = view},
            wgpu::BindGroupEntry{.binding = 1, .sampler = sampler_},
        };

        const std::string group_label
            = std::format(LOG_ID " bind group for lod {}", pass);
        const wgpu::BindGroupDescriptor group_desc{
            .label = group_label.c_str(),
            .layout = bind_group_layouts_.at(1),
            .entryCount = entries.size(),
            .entries = entries.data(),
        };

        auto create = [&] {
            bind_groups_.at(pass).at(1)
                = gpu_->getDevice().CreateBindGroup(&group_desc);
        };

        awaiter.addCall(std::move(create),
                        std::format("create bg for pass {}", pass));
    }

    return awaiter.execute();
}

std::expected<void, std::string> GenerateFeaturesPass::writeBRIEFvalues() {
    spdlog::info(LOG_ID " writing binary test values for BRIEF descriptors");

    assert(brief_tests_binding_);

    const auto device = gpu_->getDevice();
    const auto queue = device.GetQueue();
    const auto encoder = device.CreateCommandEncoder();

    constinit static const gpumodels::BRIEFTestSet tests{
#include "brief_tests.inc"
    };

    if (auto err
        = gpu_->fillNonInputBuffer(encoder, std::as_bytes(std::span(tests)),
                                   brief_tests_binding_.value())) {
        return std::unexpected("writing instructions for filling: "
                               + err.value());
    }

    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    std::string error_msg;

    auto wait_submission = [&] -> wgpu::Future {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView sv,
               std::string* err) {
                spdlog::info(
                    LOG_ID
                    " finished writing BRIEF test values. status:{} msg:'{}'",
                    static_cast<int>(status), static_cast<std::string>(sv));
                if (status != wgpu::QueueWorkDoneStatus::Success) {
                    *err = static_cast<std::string>(sv);
                }
            },
            &error_msg);
    };

    const auto res = gpu_->getAwaiter()
                         .addCall(std::move(wait_submission),
                                  "Wait writing on the GPU", false)
                         .execute();

    if (error_msg.length() > 0) {
        spdlog::error(
            LOG_ID
            " Error message when writing BRIEF values: '{}', binding: {}",
            error_msg, brief_tests_binding_.value());
    }

    if (!res) {
        return std::unexpected("gpu: " + std::move(res).error());
    }

    return {};
}

std::optional<std::string> GenerateFeaturesPass::execute() {}
