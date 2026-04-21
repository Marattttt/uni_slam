#include "vizualize_data.hpp"

#include <algorithm>
#include <cstring>
#include <flat_map>
#include <numeric>
#include <vector>

#include "common.hpp"

using namespace wslam;
using namespace wslam::viz;

#define LOG_ID "[Wgpu Resource Provider]"

namespace {
std::vector<uint32_t> bytesToUint32(std::span<const std::byte> bytes) {
    if (bytes.size() % 4 != 0) {
        throw std::invalid_argument("size must be multiple of 4");
    }

    std::vector<uint32_t> result(bytes.size() / 4);

    for (size_t i = 0; i < result.size(); ++i) {
        std::array<std::byte, 4> chunk;
        auto to_copy = bytes.subspan(i * 4, 4);

        std::ranges::copy(to_copy, chunk.begin());

        result[i] = std::bit_cast<uint32_t>(chunk);
    }

    return result;
}
};  // namespace

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

    // GET FEATURES
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

        const auto features_data = bytesToUint32(data.value());
        auto extracted = impl::ExtractFeaturePerLod(features_data);

        if (extracted) {
            for (const auto& [lod, features] : extracted.value().features) {
                result[lod].features = std::move(features);
            }

        } else {
            return std::unexpected(
                std::format("getting features: {}", extracted.error()));
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

std::expected<impl::FeaturesPerLod, std::string> impl::ExtractFeaturePerLod(
    const std::vector<uint32_t>& data) {
    const auto width = GPUConst::frame_width;
    const auto height = GPUConst::frame_height;

    const std::span data_sp(data);

    FeaturesPerLod result;

    for (uint8_t lod = 0; lod < GPUConst::levels_of_detail; ++lod) {
        const auto offset = lod * width * height;
        const auto segment
            = data_sp.subspan(offset, static_cast<size_t>(width) * height);

        // Likely cheaper than to reallocate
        const auto feature_count = static_cast<size_t>(
            std::ranges::count_if(segment, [](uint32_t el) { return el > 0; }));

        std::vector<Feature> features;
        features.reserve(feature_count);

        for (size_t i = 0; i < segment.size(); ++i) {
            if (segment[i] == 0) {
                continue;
            }

            const auto x = static_cast<uint32_t>(i % width);
            const auto y = static_cast<uint32_t>(i / width);
            const auto strength = static_cast<uint32_t>(segment[i]);

            features.emplace_back(x, y, strength);
        }

        result.features.insert({LOD{lod}, std::move(features)});
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
