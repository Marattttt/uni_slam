#pragma once

#include <webgpu/webgpu_cpp.h>

#include <flat_map>
#include <initializer_list>
#include <memory>
#include <utility>

#include "anybag.hpp"
#include "common.hpp"
#include "gpu.hpp"
#include "pass.hpp"
#include "viz.hpp"

namespace wslam::viz {

struct FeatureStyle {
    std::array<uint8_t, 3> color;
    uint8_t thickness;
};

struct Feature {
    uint32_t x;
    uint32_t y;
    uint32_t strength;
};

struct Resource {
    std::string title;
    std::optional<wslam::compute::TextureData> texture;
    std::optional<FeatureStyle> feature_style;
    std::optional<std::vector<Feature>> features;
};

constexpr FeatureStyle kDefaultFeatureStyle{.color = {255, 0, 0},
                                            .thickness = 5};
constexpr auto kDefaultTexutreName = "viz_texture";

class ReourceBuilder {
   public:
    constexpr ReourceBuilder& SetFeatureStyle(FeatureStyle style) {
        res_.feature_style = style;
        return *this;
    }
    constexpr ReourceBuilder& SetFeatures(std::vector<Feature>&& features) {
        res_.features = std::move(features);
        return *this;
    }
    constexpr ReourceBuilder& SetTexture(wslam::compute::TextureData texture) {
        res_.texture = std::move(texture);
        return *this;
    }
    constexpr ReourceBuilder& SetTitle(std::string title) {
        res_.title = std::move(title);
        return *this;
    }

    [[nodiscard]] constexpr Resource Build() const { return res_; }

   private:
    Resource res_{
        .title = ResourceIdentifier::GetFrameName(0),
        .texture = std::nullopt,
        .feature_style = kDefaultFeatureStyle,
        .features = std::nullopt,
    };
};

namespace impl {
struct FeaturesPerLod {
    std::flat_map<LOD, std::vector<Feature>> features;
};

std::expected<FeaturesPerLod, std::string> ExtractFeaturePerLod(
    const std::vector<uint32_t>& data);
};  // namespace impl

class ResourceProvider {
   public:
    using ResourceVec = std::vector<Resource>;
    [[nodiscard]] virtual std::expected<ResourceVec, std::string> GetResources()
        = 0;

    virtual ~ResourceProvider() = default;
};

class WgpuResourceProvider : public ResourceProvider {
   public:
    struct Opts {
        AnyBag& storage;
        std::shared_ptr<compute::GPU> gpu;
        std::initializer_list<LOD> lod_levels;
        std::optional<std::string> features_label = std::nullopt;
    };

    WgpuResourceProvider(Opts opts)
        : ResourceProvider(),
          storage_(opts.storage),
          gpu_(std::move(opts.gpu)),
          lod_levels_(opts.lod_levels),
          features_label_(std::move(opts.features_label)) {}

    std::expected<ResourceVec, std::string> GetResources() override;

   private:
    AnyBag& storage_;
    std::shared_ptr<compute::GPU> gpu_;
    std::initializer_list<LOD> lod_levels_;
    std::optional<std::string> features_label_;

    [[nodiscard]] std::expected<compute::TextureData, std::string> loadTexture(
        const std::string& name);

    [[nodiscard]] ResourceVec resourceMapToVec(
        std::flat_map<LOD, Resource>& features);
};

class VisualizeDataPass : public wslam::compute::Pass {
   public:
    VisualizeDataPass(const std::shared_ptr<wslam::compute::GPU>& gpu,
                      std::unique_ptr<ResourceProvider> res_provider)
        : Pass(gpu), res_provider_(std::move(res_provider)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::optional<wslam::viz::VizGUI> gui_;
    std::unique_ptr<ResourceProvider> res_provider_;

    std::optional<std::string> drawResource(Resource res);
    std::optional<std::string> drawCorners();
    void handleNext();
};
};  // namespace wslam::viz
