#include "load_features_cpu.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ranges>

#include "common.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[Load Features pass]"

std::string LoadDataCPUPass::getId() const { return LOG_ID; }

std::optional<std::string> LoadDataCPUPass::initialize() {
    spdlog::info(LOG_ID " Initializing");

    if (auto err = initBindings()) {
        return "bindings: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> LoadDataCPUPass::initBindings() {
    spdlog::info(LOG_ID " Taking bindings from storage");

    assert(!features_binding_ && !textures_);

    auto& storage = shared_.getStorage();

    auto features = storage.getPtr<compute::BufferBinding>(kFeaturesGPULabel);

    if (!features) {
        return "could not get binding for features";
    }

    features_binding_.emplace(features.value());

    const auto textures
        = std::views::iota(0U, kLodCount) | std::views::transform([&](auto i) {
              return shared_.getTexture(i);
          });

    textures_.emplace();

    std::ranges::copy(textures, textures_->begin());

    return {};
}

std::optional<std::string> LoadDataCPUPass::execute() {
    spdlog::info(LOG_ID " Exeucuting");

    auto textures = loadTextureData().transform_error(
        [](auto&& err) { return "textures: " + err; });

    if (!textures) {
        return textures.error();
    }

    spdlog::info(LOG_ID " Loaded texture data; texture sizes: {}",
                 textures.value() | std::views::transform([](auto&& t) {
                     return std::make_pair(t.width, t.height);
                 }));

    auto features = loadFeatures().transform_error(
        [](auto&& err) { return "features: " + err; });

    if (!features) {
        return features.error();
    }

    spdlog::info(LOG_ID " Loaded features data; total features: {}",
                 features.value().size());

    auto& storage = shared_.getStorage();

    for (uint32_t i = 0; i < features->size(); i++) {
        storage.set(ResourceIdentifier::GetProcessedFrameName(0, i),
                    std::move(textures).value());
    }

    shiftFeatureSets();

    storage.set(ResourceIdentifier::GetFeatureSetName(0),
                std::move(features).value());

    return {};
}

void LoadDataCPUPass::shiftFeatureSets() {
    for (auto i : std::views::iota(0U, GPUConst::featuesets_stored)
                      | std::views::reverse) {
        auto& storage = shared_.getStorage();

        auto set = storage.take<FeatureSet>(
            ResourceIdentifier::GetFeatureSetName(i), false);

        // Remove longest kept feature set
        if (!set) {
            continue;
        }

        if (i == GPUConst::featuesets_stored) {
            continue;
        }

        storage.set(ResourceIdentifier::GetFeatureSetName(i + 1),
                    std::move(set).value());
    }
}

auto LoadDataCPUPass::loadTextureData()
    -> std::expected<std::array<compute::TextureData, kLodCount>, std::string> {
    assert(textures_);

    std::array<compute::TextureData, kLodCount> result;

    for (const auto lod : std::views::iota(0UZ, kLodCount)) {
        const auto& texture = shared_.getTexture(lod);

        auto data = gpu_->readTexture(texture, GPUConst::pixel_size, true);
        if (!data) {
            return std::unexpected(std::format("reading lod {}: {}", lod,
                                               std::move(data).error()));
        }
        result.at(lod) = std::move(data).value();
    }

    return result;
}

std::expected<FeatureSet, std::string> LoadDataCPUPass::loadFeatures() {
    assert(features_binding_);

    auto data = gpu_->readBuffer(*features_binding_.value());
    if (!data) {
        return std::unexpected("reading features binding: "
                               + std::move(data).error());
    }

    const auto* bytes = data.value().data();

    const auto* features
        = reinterpret_cast<const gpumodels::FeatureArray*>(bytes);

    FeatureSet result;

    for (const auto& lod : *features) {
        for (const auto& f : lod.values | std::views::take(lod.count)) {
            result.at(f.lod).emplace_back(f);
        }
    }

    return result;
}
