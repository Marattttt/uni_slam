#include "vizualize_data.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <expected>
#include <flat_map>
#include <memory>
#include <span>
#include <vector>

#include "common.hpp"
#include "compute.hpp"
#include "gpu.hpp"
#include "models.hpp"

using namespace wslam;
using namespace wslam::viz;

#define LOG_ID "[Wgpu Resource Provider]"

namespace {
namespace utils {
std::flat_map<size_t, std::vector<Feature>> ExtractFeatures(
    std::span<const std::byte> data);
};
}  // namespace

std::expected<WgpuResourceProvider::ResourceVec, std::string>
WgpuResourceProvider::GetResources() {
    if (matches_label_) {
        return std::unexpected(
            "feature matching is not supported by WgpuResourceProvider; "
            "use CpuResourceProvider instead");
    }

    std::flat_map<size_t, Resource> result;

    const auto debug_check_zero_values
        = [](auto data, auto is_zero, std::string_view label) {
#ifndef NDEBUG
              uint32_t nonzero = 0;
              for (const auto& vec : data) {
                  nonzero += std::ranges::count_if(vec, is_zero);
              }

              if (nonzero == 0) {
                  spdlog::warn(LOG_ID " all zero values for {}", label);
              }
#endif
          };

    // GET TEXTURES AND CREATE ENTRIES
    for (const auto lod : lod_levels_) {
        result.insert(
            {lod, Resource{
                      .title = std::format("Vizualization for lod {}", lod.v),
                      .texture = {},
                      .corner_style = kDefaultCornerStyle,
                      .corners = std::nullopt,
                      .feature_style = kDefaultFeatureStyle,
                      .features = std::nullopt,
                      .brief_tests = kDefaultBRIEFTestSet,
                      .match_style = std::nullopt,
                      .feature_matches = std::nullopt,
                  }});

        auto& res = result.at(LOD{lod});

        if (auto texture = loadTexture(lod)) {
            res.texture = std::move(texture.value());
        } else {
            return std::unexpected("getting texture: " + texture.error());
        }
    }

    if (corners_label_) {
        const auto corners
            = storage_.getPtr<compute::BufferBinding>(corners_label_.value());
        if (!corners) {
            return std::unexpected("could not get corners binding");
        }

        const auto data = gpu_->readBuffer(**corners);
        if (!data) {
            return std::unexpected("reading corners data: "
                                   + std::move(data).error());
        }

        auto extracted = impl::ExtractLodCorners(data.value());

        debug_check_zero_values(
            std::span(extracted.values()),
            [](const Corner& c) { return c.strength > 0; }, "corners");

        for (const auto& [lod, features] : extracted) {
            result[lod].corners = std::move(features);
        }
    }

    if (features_label_) {
        const auto features
            = storage_.getPtr<compute::BufferBinding>(*features_label_);

        if (!features) {
            return std::unexpected("could not get features binding");
        }

        const auto data = gpu_->readBuffer(**features);

        if (!data) {
            return std::unexpected("reading features data: "
                                   + std::move(data).error());
        }

        auto extracted = utils::ExtractFeatures(data.value());

        debug_check_zero_values(
            std::span(extracted.values()),
            [](const Feature& c) { return c.strength > 0; }, "features");

        for (const auto& [lod, features] : extracted) {
            result[lod].features = std::move(features);
        }
    }

    return std::move(result).extract().values;
}

std::expected<compute::TextureData, std::string>
WgpuResourceProvider::loadTexture(size_t lod) {
    const auto& texture = shared_.getTexture(lod);

    const auto texture_data
        = gpu_->readTexture(texture, GPUConst::pixel_size, true)
              .transform_error([](const std::string& err) {
                  return "reading texture: " + err;
              });

    return texture_data;
}

impl::FeaturesPerLod impl::ExtractLodCorners(std::span<const std::byte> data) {
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

std::flat_map<size_t, std::vector<Feature>> utils::ExtractFeatures(
    std::span<const std::byte> data) {
    std::flat_map<size_t, std::vector<Feature>> result;

    const auto* array
        = reinterpret_cast<const gpumodels::FeatureArray<>*>(data.data());

    std::span<const Feature> features{array->values.data(), array->count};

    for (const auto& feat : features) {
        result[feat.lod].emplace_back(feat);
    }

    return result;
}

#undef LOG_ID
#define LOG_ID "[CPU Resource Provider]"

std::expected<CpuResourceProvider::ResourceVec, std::string>
CpuResourceProvider::GetResources() {
    using TextureArray = std::array<compute::TextureData, kLodCount>;

    const auto textures_ptr = storage_.getPtr<TextureArray>(
        ResourceIdentifier::GetProcessedFrameName(0, 0));
    if (!textures_ptr) {
        return std::unexpected("could not get texture array from storage");
    }
    const auto& textures = **textures_ptr;

    const FeatureSet* features_ptr = nullptr;
    if (load_features_) {
        const auto ptr = storage_.getPtr<FeatureSet>(
            ResourceIdentifier::GetFeatureSetName(0));
        if (!ptr) {
            return std::unexpected("could not get feature set from storage");
        }
        features_ptr = *ptr;
    }

    const MatchResult* matches_ptr = nullptr;
    if (load_matches_) {
        const auto ptr = storage_.getPtr<MatchResult>(
            ResourceIdentifier::MatchedFeaturesName);
        if (!ptr) {
            return std::unexpected(
                "could not get feature match result from storage");
        }
        matches_ptr = *ptr;
    }

    ResourceVec result;

    for (const auto lod : lod_levels_) {
        if (lod.v >= kLodCount) {
            return std::unexpected(
                std::format("LOD {} is out of range (max {})", lod.v,
                            kLodCount - 1));
        }

        Resource res{
            .title = std::format("CPU Visualization for lod {}", lod.v),
            .texture = textures.at(lod.v),
            .corner_style = std::nullopt,
            .corners = std::nullopt,
            .feature_style = kDefaultFeatureStyle,
            .features = std::nullopt,
            .brief_tests = kDefaultBRIEFTestSet,
            .match_style = std::nullopt,
            .feature_matches = std::nullopt,
        };

        if (features_ptr != nullptr) {
            res.features = features_ptr->at(lod.v);
        }

        result.push_back(std::move(res));
    }

    if (matches_ptr != nullptr) {
        const auto& lod0_texture = textures.at(0);

        std::vector<FeaturePair> pairs;
        for (const auto& lod_map : *matches_ptr) {
            for (const auto& [curr, prev] : lod_map) {
                pairs.emplace_back(prev, curr);
            }
        }

        if (!pairs.empty()) {
            result.push_back(Resource{
                .title = "Feature Matches",
                .texture = lod0_texture,
                .corner_style = std::nullopt,
                .corners = std::nullopt,
                .feature_style = std::nullopt,
                .features = std::nullopt,
                .brief_tests = std::nullopt,
                .match_style = kDefaultMatchStyle,
                .feature_matches = std::move(pairs),
            });
        }
    }

    return result;
}

#undef LOG_ID
#define LOG_ID "[Vizualize Data pass]"

std::string VisualizeDataPass::getId() const { return LOG_ID; }

std::optional<std::string> VisualizeDataPass::initialize() {
    spdlog::info(LOG_ID " Initializing");
    auto gui_res = VizGUI::create({}).transform_error(
        [](auto&& err) { return "creating window: " + err; });

    if (!gui_res) {
        return std::move(gui_res).error();
    }

    gui_ = std::move(gui_res.value());
    initCallbacks();

    return {};
}

void VisualizeDataPass::initNextResourceCallback() {
    gui_->addCallback('n', [this] {
        if (resource_count_ > 0) {
            current_idx_ = (current_idx_ + 1) % resource_count_;
            spdlog::debug(LOG_ID " advancing to resource {}", current_idx_);
        }
    });
}

void VisualizeDataPass::initCallbacks() {
    initNextResourceCallback();

    gui_->addCallback('N', [this] { advance_frame_ = true; });

    for (char c = '1'; c <= '9'; c++) {
        gui_->addCallback(c, [this, c] {
            frames_to_skip_ = (c - '0') * 5 - 1;
            advance_frame_ = true;
        });
    }
    gui_->addCallback('0', [this] {
        frames_to_skip_ = 49;
        advance_frame_ = true;
    });
}

std::optional<std::string> VisualizeDataPass::execute() {
    if (frames_to_skip_ > 0) {
        frames_to_skip_--;
        return std::nullopt;
    }

    auto resources = res_provider_->GetResources();
    if (!resources) {
        return "getting data: " + resources.error();
    }

    if (resources->empty()) {
        spdlog::warn(LOG_ID " No resources to visualize");
        return std::nullopt;
    }

    resource_count_ = resources->size();
    current_idx_ = 0;
    advance_frame_ = false;

#ifndef NDEBUG
    const auto& res = resources.value();
    for (auto i = 0UZ; i < res.size(); ++i) {
        spdlog::debug(LOG_ID " Resource {}: {}", i, res.at(i));
    }
#endif

    while (!gui_->windowShouldClose() && !advance_frame_) {
        gui_->startFrame();

        const auto& resource = resources->at(current_idx_);
        if (auto err = drawResource(resource)) {
            return "drawing " + resource.title + ": " + err.value();
        }

        gui_->endFrame();
    }

    if (gui_->windowShouldClose()) {
        return compute::kFullStopExecution;
    }

    return std::nullopt;
}

namespace {
VizTexture textureDataToVizTexture(compute::TextureData&& texture) {
    assert(texture.pixel_data_rgb.size() % 3 == 0);

    const auto& data = texture.pixel_data_rgb;

    std::vector<data::PixelRGB> pixels;
    pixels.reserve(texture.pixel_data_rgb.size() / 3);

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

    VizTexture texture
        = textureDataToVizTexture(std::move(res.texture).value());
    gui_->drawTexture(texture);

    if (res.corners.has_value() && !res.corners->empty()) {
        drawCorners(res);
    }

    if (res.features.has_value() && !res.features->empty()) {
        drawFeatures(res);
    }

    if (res.feature_matches.has_value() && !res.feature_matches->empty()) {
        drawMatches(res);
    }

    return std::nullopt;
}

void VisualizeDataPass::drawCorners(const Resource& res) {
    const CornerStyle style = res.corner_style.value_or(kDefaultCornerStyle);

    std::vector<VizCorner> viz_features;
    viz_features.reserve(res.corners->size());
    for (const auto& f : res.corners.value()) {
        viz_features.push_back(
            {.x = static_cast<float>(f.x), .y = static_cast<float>(f.y)});
    }

    // Use `thickness` as a visual radius in pixels. If you want
    // strength to modulate radius, scale per-feature here instead.
    const float radius = static_cast<float>(style.thickness) * 1.5F;
    gui_->drawCorners({.corners = viz_features,
                       .color = style.color,
                       .radius = radius,
                       .image_width = res.texture.value().width,
                       .image_height = res.texture.value().height});
}

void VisualizeDataPass::drawFeatures(const Resource& res) {
    const auto style = res.feature_style.value_or(kDefaultFeatureStyle);

    std::vector<Feature> features;
    features.reserve(res.features.value().size());

    gui_->drawFeatures({
        .features = res.features.value(),
        .test_set = res.brief_tests.value_or(kDefaultBRIEFTestSet),
        .feature_color = style.feature_color,
        .bit_one_color = style.bit_one_color,
        .bit_zero_color = style.bit_zero_color,
        .radius = style.radius,
        .image_width = res.texture.value().width,
        .image_height = res.texture.value().height,
    });
}

void VisualizeDataPass::drawMatches(const Resource& res) {
    const auto style = res.match_style.value_or(kDefaultMatchStyle);
    const auto& pairs = res.feature_matches.value();

    const float img_w = static_cast<float>(res.texture.value().width);
    const float img_h = static_cast<float>(res.texture.value().height);

    auto lod_scale = [](uint32_t lod) -> float {
        float scale = 1.0F;
        for (uint32_t i = 0; i < lod; ++i) {
            scale *= static_cast<float>(GPUConst::lod_scale_factor);
        }
        return scale;
    };

    std::vector<VizMatchLine> match_lines;
    match_lines.reserve(pairs.size());
    for (const auto& [prev, curr] : pairs) {
        const float scale_prev = lod_scale(prev.lod);
        const float scale_curr = lod_scale(curr.lod);
        const float ax = static_cast<float>(prev.x) * scale_prev;
        const float ay = static_cast<float>(prev.y) * scale_prev;
        const float bx = static_cast<float>(curr.x) * scale_curr;
        const float by = static_cast<float>(curr.y) * scale_curr;

        if (ax < 0 || ax > img_w || ay < 0 || ay > img_h
            || bx < 0 || bx > img_w || by < 0 || by > img_h) {
            spdlog::warn(LOG_ID " match out of image bounds [{:.0f}x{:.0f}], "
                         "skipping: prev({:.1f},{:.1f}) curr({:.1f},{:.1f})",
                         img_w, img_h, ax, ay, bx, by);
            continue;
        }

        match_lines.push_back({
            .a = {.x = ax, .y = ay},
            .b = {.x = bx, .y = by},
        });
    }

    gui_->drawMatches({
        .matches = match_lines,
        .line_color = style.line_color,
        .point_a_color = style.point_a_color,
        .point_b_color = style.point_b_color,
        .radius = style.radius,
        .image_width = res.texture.value().width,
        .image_height = res.texture.value().height,
    });
}
