#include "vizualize_data.hpp"

#include <cstring>
#include <flat_map>
#include <memory>
#include <vector>

#include "common.hpp"
#include "models.hpp"

using namespace wslam;
using namespace wslam::viz;

#define LOG_ID "[Wgpu Resource Provider]"

std::expected<WgpuResourceProvider::ResourceVec, std::string>
WgpuResourceProvider::GetResources() {
    std::flat_map<LOD, Resource> result;

    // GET TEXTURES AND CREATE ENTRIES
    for (const auto lod : lod_levels_) {
        result.insert(
            {lod, Resource{
                      .title = std::format("Vizualization for lod {}", lod.v),
                      .texture = {},
                      .feature_style = kDefaultFeatureStyle,
                      .features = std::nullopt,
                  }});

        auto& res = result.at(LOD{lod});

        const std::string texture_name
            = ResourceIdentifier::GetFrameName({0, LOD{lod}});

        if (auto texture = loadTexture(texture_name)) {
            res.texture = std::move(texture.value());
        } else {
            return std::unexpected("getting texture: " + texture.error());
        }
    }

    if (features_label_) {
        const auto features_bind
            = storage_.getPtr<compute::BufferBinding>(features_label_.value());
        if (!features_bind) {
            return std::unexpected("could not get features binding");
        }

        auto data = gpu_->readBuffer(**features_bind);
        if (!data) {
            return std::unexpected("reading feautres buffer binding: "
                                   + data.error());
        }

        auto extracted = impl::ExtractFeaturePerLod(data.value());

        for (const auto& [lod, features] : extracted) {
            result[lod].features = std::move(features);
        }
    }

    return std::move(result).extract().values;
}

std::expected<compute::TextureData, std::string>
WgpuResourceProvider::loadTexture(const std::string& name) {
    const auto texture_ptr = storage_.getPtr<wgpu::Texture>(name);
    if (!texture_ptr) {
        return std::unexpected("could not get gexture by key");
    }
    const auto& texture = *texture_ptr.value();

    const auto texture_data
        = gpu_->readTexture(texture, GPUConst::pixel_size, true)
              .transform_error([](const std::string& err) {
                  return "reading texture: " + err;
              });

    return texture_data;
}

impl::FeaturesPerLod impl::ExtractFeaturePerLod(
    std::span<const std::byte> data) {
    using Block = gpumodels::CornersBlock<GPUConst::frame_width,
                                          GPUConst::frame_height>;

    const size_t blocks_cnt = data.size_bytes() / sizeof(Block);

    const std::span<const Block> blocks(
        reinterpret_cast<const Block*>(data.data()), blocks_cnt);

    FeaturesPerLod result;

    for (size_t i = 0; i < blocks.size(); i++) {
        const auto lod = blocks[i];
        std::vector<Corner> features;

        for (size_t x = 0; x < lod.width; x++) {
            for (size_t y = 0; y < lod.height; y++) {
                const auto str = lod[x, y];
                if (str > 0) {
                    features.emplace_back(x, y, str);
                }
            }
        }

        result.insert_or_assign(LOD{static_cast<uint8_t>(i)},
                                std::move(features));
    }

    return result;
}

#undef LOG_ID
#define LOG_ID "[Vizualize Data pass]"

std::string VisualizeDataPass::getId() const { return LOG_ID; }

std::optional<std::string> VisualizeDataPass::initialize() {
    return std::nullopt;
}

std::optional<std::string> VisualizeDataPass::execute() {
    gui_ = std::move(VizGUI::create({}).value());

    auto resources = res_provider_->GetResources();
    if (!resources) {
        return "getting data: " + resources.error();
    }

    if (resources->empty()) {
        spdlog::warn(LOG_ID " No resources to visualize");
        return std::nullopt;
    }

#ifndef NDEBUG
    const auto& res = resources.value();
    for (size_t lod = 0; lod < res.size(); lod++) {
        spdlog::debug(LOG_ID " Resources for lod:{}: total_features:{}", lod,
                      res.at(lod).features.value_or({}).size());

        const size_t pairs_to_print = 30;
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> to_print;
        to_print.reserve(pairs_to_print);

        for (const auto& feature :
             res.at(lod).features.value() | std::views::take(pairs_to_print)) {
            to_print.emplace_back(feature.x, feature.y, feature.strength);
        }

        spdlog::debug(LOG_ID " First enties: {}", to_print);
    }
#endif

    size_t current_idx = 0;
    gui_->addRequestNextCallback([&] {
        current_idx = (current_idx + 1) % resources->size();
        spdlog::debug(LOG_ID " advancing to resource {}", current_idx);
    });

    while (!gui_->windowShouldClose()) {
        gui_->startFrame();

        const auto& resource = resources->at(current_idx);
        if (auto err = drawResource(resource)) {
            return "drawing " + resource.title + ": " + err.value();
        }

        gui_->endFrame();
    }
    return std::nullopt;
}

namespace {
VizTexture textureDataToVizTexture(compute::TextureData&& texture) {
    assert(texture.pixel_data.size() % 3 == 0);

    const auto& data = texture.pixel_data;

    std::vector<data::PixelRGB> pixels;
    pixels.reserve(texture.pixel_data.size() / 3);

    for (size_t i = 0; i < data.size() - 2; i += 3) {
        pixels.emplace_back(static_cast<uint8_t>(data[i]),
                            static_cast<uint8_t>(data[i + 1]),
                            static_cast<uint8_t>(data[i + 2]));
    }

    return VizTexture{
        .width = texture.width,
        .height = texture.height,
        .data = std::move(pixels),
    };
}
};  // namespace

std::optional<std::string> VisualizeDataPass::drawResource(Resource res) {
    if (!res.texture.has_value()) {
        return std::nullopt;  // nothing to show for this resource
    }

    auto& frame = res.texture.value();

    // Adjust the field accesses below to match your FrameRGB definition.
    VizTexture texture = textureDataToVizTexture(std::move(frame));
    gui_->drawTexture(texture);

    if (res.features.has_value() && !res.features->empty()) {
        const FeatureStyle style
            = res.feature_style.value_or(kDefaultFeatureStyle);

        std::vector<VizFeature> viz_features;
        viz_features.reserve(res.features->size());
        for (const auto& f : *res.features) {
            viz_features.push_back(
                {.x = static_cast<float>(f.x), .y = static_cast<float>(f.y)});
        }

        // Use `thickness` as a visual radius in pixels. If you want
        // strength to modulate radius, scale per-feature here instead.
        const float radius = static_cast<float>(style.thickness) * 1.5F;
        gui_->drawFeatures({.features = viz_features,
                            .color = style.color,
                            .radius = radius,
                            .image_width = texture.width,
                            .image_height = texture.height});
    }

    return std::nullopt;
}
