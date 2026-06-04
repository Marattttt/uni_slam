#include "generate_features.hpp"

#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <expected>
#include <ranges>
#include <span>
#include <type_traits>

#include "common.hpp"
#include "gpu.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[Generate Descriptors pass]"

namespace {
namespace CONSTANTS {
constexpr std::string_view kShaderPath = "generate_features.wgsl";
constexpr auto kLodValuesBindingSize = 256 * GPUConst::levels_of_detail;
constexpr auto kBriefTestsBindingSize
    = sizeof(std::array<std::array<int32_t, 4>, 256>);
constexpr auto kCornersBindingSize = AddPadding(
    sizeof(
        gpumodels::CornersBlock<GPUConst::frame_width, GPUConst::frame_height>)
        * GPUConst::levels_of_detail,
    256);
// Element count of a single LoD's strengths array in the shader — NOT a byte
// size. Must match the producers (detect_corners / cull_corners overrides).
constexpr auto kCornersPerLod = GPUConst::frame_width * GPUConst::frame_height;
constexpr auto kFeaturesBindingSize
    = AddPadding(sizeof(gpumodels::FeatureArray), 256);

constexpr size_t kWgSize = 8;
constexpr std::array<wgpu::ConstantEntry, 2> kShaderOverrides{
    wgpu::ConstantEntry{
        .key = "WG_SIZE_X",
        .value = 8,
    },
    wgpu::ConstantEntry{
        .key = "WG_SIZE_Y",
        .value = 8,
    },
};

std::array<compute::GPU::ShaderOverride, 3> kTextShaderOverrides{
    compute::GPU::ShaderOverride{
        "LOD_COUNT",
        std::format("{}u", GPUConst::levels_of_detail),
    },
    compute::GPU::ShaderOverride{
        "CORNER_BLOCK_SZ",
        std::format("{}u", kCornersPerLod),
    },
    compute::GPU::ShaderOverride{
        "FEATURE_BLOCK_SZ",
        std::format("{}u", GPUConst::max_features_per_lod),

    },
};

};  // namespace CONSTANTS
};  // namespace

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
    if (auto err = label(writeConstantValues(), "write constant values")) {
        return err;
    }

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

    return gpu_->getAwaiter().runChecked(create, "Create sampler").execute();
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

    const std::array<wgpu::BindGroupLayoutEntry, 4> entries{
        // @group(0) @binding(0) var<storage, read> corners: array<CornerBlock,
        // LOD_COUNT>;
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
        },
        //  @group(0) @binding(1) var<uniform> brief_tests: array<PointPair,
        //  256>;
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Uniform},
        },
        // @group(0) @binding(2) var<storage, read_write> features:
        // FeatureArray;
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        },
        // @group(0) @binding(3) var samp: sampler; // linear filtering,
        // clamp-to-edge
        wgpu::BindGroupLayoutEntry{
            .binding = 3,
            .visibility = wgpu::ShaderStage::Compute,
            .sampler = {.type = wgpu::SamplerBindingType::Filtering},
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
        .runChecked(create, "Create layout for common bind group")
        .execute();
}

std::expected<void, std::string>
GenerateFeaturesPass::initPerPassBindgroupLayout() {
    spdlog::debug(LOG_ID " Initializing per pass bind group");

    const std::array<wgpu::BindGroupLayoutEntry, 2> entries{
        // @group(1) @binding(0) var image: texture_2d<f32>;
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Compute,
            .texture = {.sampleType = wgpu::TextureSampleType::Float,
                        .viewDimension = wgpu::TextureViewDimension::e2D},
        },
        // @group(1) @binding(1) var<storage, read> lod: u32;
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type = wgpu::BufferBindingType::Uniform,
                       .hasDynamicOffset = true,
                       .minBindingSize = sizeof(uint32_t)},
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
        .runChecked(create, "Create layout for per pass bind group")
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

    auto* const corners_binding
        = shared_.getStorage()
              .getPtr<compute::BufferBinding>(kCornersLabel)
              .value();

    using BT = compute::BufferType;

    const auto features_storage_type
        = corners_binding->getBuffertype() == BT::StorageA ? BT::StorageB
                                                           : BT::StorageA;

    spdlog::debug("left {} right {}", CONSTANTS::kCornersBindingSize,
                  corners_binding->getSize());
    assert(CONSTANTS::kCornersBindingSize == corners_binding->getSize());

    std::array<wgpu::BindGroupEntry, 4> entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = gpu_->getBuffer(corners_binding->getBuffertype()),
            .offset = corners_binding->getOffset(),
            .size = CONSTANTS::kCornersBindingSize},
        wgpu::BindGroupEntry{.binding = 1,
                             .size = CONSTANTS::kBriefTestsBindingSize},
        wgpu::BindGroupEntry{.binding = 2,
                             .size = CONSTANTS::kFeaturesBindingSize},
        wgpu::BindGroupEntry{.binding = 3, .sampler = sampler_},
    };

    auto bindings
        = gpu_->assignBuffersAndOffsets(compute::GPU::BindGroupBindings{
            {
                kBriefTestsBindingLabel,
                compute::GPU::BgBinding{
                    .buf_type = compute::BufferType::Uniform,
                    .bg_entry = entries.at(1),
                    .is_retained = true},
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
        .runChecked(create, "create bind group")
        .execute()
        .transform_error(
            [](const auto& err) { return "creating bind groups: " + err; });
}

std::expected<void, std::string> GenerateFeaturesPass::initPerPassBindgroups() {
    spdlog::debug(LOG_ID " Initializing per pass bind groups");

    // Allocate the LOD-values region ONCE — the dynamic offset at dispatch
    // time picks which 256-byte slot inside it the shader sees.
    wgpu::BindGroupEntry lod_entry{
        .binding = 1,
        .size = CONSTANTS::kLodValuesBindingSize,
    };
    auto lod_bindings
        = gpu_->assignBuffersAndOffsets(compute::GPU::BindGroupBindings{
            {kLodValuesBindingLabel,
             {.buf_type = compute::BufferType::Uniform,
              .bg_entry = lod_entry,
              .is_retained = true}},
        });
    if (!lod_bindings) {
        return std::unexpected("assigning binding for LOD idxs: "
                               + lod_bindings.error());
    }
    lod_idxs_binding_
        = std::move(lod_bindings.value().at(kLodValuesBindingLabel));

    auto awaiter = gpu_->getAwaiter();

    for (size_t lod = 0; lod < kPassCount; lod++) {
        const auto texture = shared_.getTexture(lod);

        const std::string view_label
            = std::format(LOG_ID " view into lod {}", lod);
        const wgpu::TextureViewDescriptor view_desc{
            .label = view_label.c_str(),
            .format = wgpu::TextureFormat::R32Float,
            .dimension = wgpu::TextureViewDimension::e2D,
            .usage = wgpu::TextureUsage::TextureBinding,
        };
        const auto view = texture.CreateView(&view_desc);

        std::array<wgpu::BindGroupEntry, 2> entries{
            wgpu::BindGroupEntry{.binding = 0, .textureView = view},
            lod_entry,  // same buffer/offset/size for every pass
        };

        const std::string group_label
            = std::format(LOG_ID " bind group for lod {}", lod);
        const wgpu::BindGroupDescriptor group_desc{
            .label = group_label.c_str(),
            .layout = bind_group_layouts_.at(1),
            .entryCount = entries.size(),
            .entries = entries.data(),
        };

        auto create = [&] {
            bind_groups_.at(lod).at(1)
                = gpu_->getDevice().CreateBindGroup(&group_desc);
        };
        awaiter.runChecked(create, std::format("create bg for pass {}", lod));
    }

    return awaiter.execute();
}

std::expected<void, std::string> GenerateFeaturesPass::initComputePipeline() {
    spdlog::info(LOG_ID " Initializing compute pipeline");

    const auto res_shader
        = gpu_->loadShaderModule(CONSTANTS::kShaderPath,
                                 LOG_ID " compute shader",
                                 std::span(CONSTANTS::kTextShaderOverrides))
              .transform_error(
                  [](auto&& err) { return "loading shader: " + err; });
    if (!res_shader) {
        return std::unexpected(std::move(res_shader).error());
    }

    const auto& shader = res_shader.value();

    const wgpu::PipelineLayoutDescriptor layout_desc{
        .label = LOG_ID " pipeline layout",
        .bindGroupLayoutCount = bind_group_layouts_.size(),
        .bindGroupLayouts = bind_group_layouts_.data(),
    };

    auto create_layout = [&] {
        compute_pipeline_layout_
            = gpu_->getDevice().CreatePipelineLayout(&layout_desc);
    };

    if (auto res
        = gpu_->getAwaiter()
              .runChecked(create_layout, "create pipeline layout")
              .execute()
              .transform_error([](auto&& err) { return "layout: " + err; });
        !res) {
        return res;
    }

    const wgpu::ComputePipelineDescriptor comp_desc{
        .label = LOG_ID " compute pipeline",
        .layout = compute_pipeline_layout_,
        .compute = {.module = shader,
                    .entryPoint = "main",
                    .constantCount = CONSTANTS::kShaderOverrides.size(),
                    .constants = CONSTANTS::kShaderOverrides.data()},
    };

    auto create_pipeline = [&] {
        compute_pipeline_ = gpu_->getDevice().CreateComputePipeline(&comp_desc);
    };

    if (auto res
        = gpu_->getAwaiter()
              .runChecked(create_pipeline, "create compute pipeline")
              .execute()
              .transform_error([](auto&& err) { return "pipeline: " + err; });
        !res) {
        return res;
    }

    return {};
}

[[nodiscard]] std::expected<void, std::string>
GenerateFeaturesPass::writeConstantValues() {
    spdlog::info(LOG_ID " writing binary test values for BRIEF descriptors");

    assert(lod_idxs_binding_);
    assert(brief_tests_binding_);

    const auto encoder = gpu_->getDevice().CreateCommandEncoder();

    constinit static const gpumodels::BRIEFTestSet tests{
#include "brief_tests.inc"
    };

    if (auto err
        = gpu_->fillNonInputBuffer(encoder, std::as_bytes(std::span(tests)),
                                   brief_tests_binding_.value())) {
        return std::unexpected("writing instructions for filling: "
                               + err.value());
    }

    struct alignas(256) PaddedUint {
        std::array<uint32_t, 8> values;
    };
    static_assert(sizeof(PaddedUint) == 256);

    const auto lod_buffer = std::views::iota(0U, kPassCount)
                            | std::views::transform([](auto&& num) {
                                  return PaddedUint{.values = {num}};
                              })
                            | std::ranges::to<std::vector<PaddedUint>>();

    if (auto err = gpu_->fillNonInputBuffer(
            encoder, std::as_bytes(std::span(lod_buffer)),
            lod_idxs_binding_.value())) {
        return std::unexpected("preparing to fill lod indexes: " + err.value());
    }

    const auto commands = encoder.Finish();

    if (auto err = gpu_->submitAndWait(commands, "write BRIEF test values")) {
        return std::unexpected("gpu: " + err.value());
    }

    return {};
}

std::optional<std::string> GenerateFeaturesPass::prepareExecute(
    const wgpu::CommandEncoder& encoder) {
    spdlog::info(LOG_ID " Executing");

    for (size_t p = 0; p < kPassCount; p++) {
        writeSinglePassCommands(encoder, p);
    }

    return std::nullopt;
}

void GenerateFeaturesPass::writeSinglePassCommands(
    const wgpu::CommandEncoder& encoder, size_t passIdx) {
    const auto pass = encoder.BeginComputePass();

    const std::string label = std::format("pass for lod {}", passIdx);

    pass.SetLabel(label.c_str());
    pass.SetPipeline(compute_pipeline_);
    pass.SetBindGroup(0, bind_groups_.at(passIdx).at(0));

    const auto lod_offset = static_cast<uint32_t>(256 * passIdx);

    pass.SetBindGroup(1, bind_groups_.at(passIdx).at(1), 1, &lod_offset);

    const auto texture = shared_.getTexture(passIdx);
    const auto width = texture.GetWidth();
    const auto height = texture.GetHeight();
    pass.DispatchWorkgroups(CeilDiv(width, CONSTANTS::kWgSize),
                            CeilDiv(height, CONSTANTS::kWgSize));

    pass.End();
}
