#pragma once

#include <webgpu/webgpu_cpp.h>

#include <flat_map>
#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>

#include "anybag.hpp"
#include "common.hpp"
#include "compute/gpu.hpp"
#include "models.hpp"
#include "compute/pass.hpp"
#include "viz.hpp"

namespace wslam::viz {

struct CornerStyle {
    std::array<uint8_t, 3> color;
    uint8_t thickness;
};

struct FeatureStyle {
    std::array<uint8_t, 3> feature_color;
    std::array<uint8_t, 3> bit_one_color;
    std::array<uint8_t, 3> bit_zero_color;
    float radius;
};

struct MatchStyle {
    std::array<uint8_t, 3> line_color;
    std::array<uint8_t, 3> point_a_color;
    std::array<uint8_t, 3> point_b_color;
    float radius;
};

// Viz-only intermediate: pre-projected landmark + a depth-mapped color.
// The pass that builds this resource is responsible for projecting each
// 3D landmark back to LOD-0 pixel coordinates and assigning a color, so
// the GUI layer stays purely 2D.
struct VizLandmark2D {
    float x;
    float y;
    std::array<uint8_t, 3> color;
    float depth;
};

struct LandmarkStyle {
    float radius;
};

struct Resource {
    std::string title;
    std::optional<wslam::compute::TextureData> texture;
    std::optional<CornerStyle> corner_style;
    std::optional<std::vector<Corner>> corners;
    std::optional<FeatureStyle> feature_style;
    std::optional<std::vector<Feature>> features;
    std::optional<std::reference_wrapper<const gpumodels::BRIEFTestSet>>
        brief_tests;
    std::optional<MatchStyle> match_style;
    std::optional<std::vector<FeaturePair>> feature_matches;
    std::optional<LandmarkStyle> landmark_style;
    std::optional<std::vector<VizLandmark2D>> landmarks;
};

constexpr CornerStyle kDefaultCornerStyle{.color = {255, 0, 0}, .thickness = 5};
constexpr FeatureStyle kDefaultFeatureStyle{
    .feature_color = {255, 0, 0},
    .bit_one_color = {255, 255, 255},
    .bit_zero_color = {0, 0, 255},
    .radius = 5.0F,
};
constexpr MatchStyle kDefaultMatchStyle{
    .line_color = {255, 255, 0},
    .point_a_color = {0, 255, 0},
    .point_b_color = {255, 0, 255},
    .radius = 4.0F,
};
constexpr MatchStyle kInlierMatchStyle{
    .line_color = {0, 255, 0},
    .point_a_color = {0, 200, 255},
    .point_b_color = {255, 128, 0},
    .radius = 4.0F,
};
constexpr LandmarkStyle kDefaultLandmarkStyle{
    .radius = 5.0F,
};
constexpr auto kDefaultTextureName = "viz_texture";
constexpr gpumodels::BRIEFTestSet kDefaultBRIEFTestSet{
#include "feature_detect/brief_tests.inc"
};

class ReourceBuilder {
   public:
    constexpr ReourceBuilder& SetFeatureStyle(CornerStyle style) {
        res_.corner_style = style;
        return *this;
    }
    constexpr ReourceBuilder& SetFeatures(std::vector<Corner>&& features) {
        res_.corners = std::move(features);
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
        .corner_style = std::nullopt,
        .corners = std::nullopt,
        .feature_style = std::nullopt,
        .features = std::nullopt,
        .brief_tests = std::nullopt,
        .match_style = std::nullopt,
        .feature_matches = std::nullopt,
        .landmark_style = std::nullopt,
        .landmarks = std::nullopt,
    };
};

namespace impl {
using FeaturesPerLod = std::flat_map<LOD, std::vector<Corner>>;
;

FeaturesPerLod ExtractLodCorners(std::span<const std::byte> data);
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
        GpuSharedBindings& shared;
        std::shared_ptr<compute::GPU> gpu;
        std::initializer_list<LOD> lod_levels;
        std::optional<std::string> corners_label = std::nullopt;
        std::optional<std::string> features_label = std::nullopt;
        std::optional<std::string> matches_label = std::nullopt;
    };

    WgpuResourceProvider(Opts opts)
        : ResourceProvider(),
          storage_(opts.storage),
          shared_(opts.shared),
          gpu_(std::move(opts.gpu)),
          lod_levels_(opts.lod_levels),
          corners_label_(std::move(opts.corners_label)),
          features_label_(std::move(opts.features_label)),
          matches_label_(std::move(opts.matches_label)) {}

    std::expected<ResourceVec, std::string> GetResources() override;

   private:
    AnyBag& storage_;
    GpuSharedBindings& shared_;
    std::shared_ptr<compute::GPU> gpu_;
    std::initializer_list<LOD> lod_levels_;
    std::optional<std::string> corners_label_;
    std::optional<std::string> features_label_;
    std::optional<std::string> matches_label_;

    [[nodiscard]] std::expected<compute::TextureData, std::string> loadTexture(
        size_t lod);

    [[nodiscard]] ResourceVec resourceMapToVec(
        std::flat_map<LOD, Resource>& features);
};

class CpuResourceProvider : public ResourceProvider {
   public:
    struct Opts {
        AnyBag& storage;
        std::initializer_list<LOD> lod_levels;
        bool load_features = false;
        bool load_matches = false;
        bool load_ransac_inliers = false;
        bool load_landmarks = false;
    };

    explicit CpuResourceProvider(Opts opts)
        : ResourceProvider(),
          storage_(opts.storage),
          lod_levels_(opts.lod_levels),
          load_features_(opts.load_features),
          load_matches_(opts.load_matches),
          load_ransac_inliers_(opts.load_ransac_inliers),
          load_landmarks_(opts.load_landmarks) {}

    std::expected<ResourceVec, std::string> GetResources() override;

   private:
    static constexpr size_t kLodCount = GPUConst::levels_of_detail;

    AnyBag& storage_;
    std::vector<LOD> lod_levels_;
    bool load_features_;
    bool load_matches_;
    bool load_ransac_inliers_;
    bool load_landmarks_;
};

class VisualizeDataPass : public wslam::compute::Pass {
   public:
    VisualizeDataPass(const std::shared_ptr<wslam::compute::GPU>& gpu,
                      std::unique_ptr<ResourceProvider> res_provider)
        : gpu_(gpu), res_provider_(std::move(res_provider)) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<compute::GPU> gpu_;

    std::optional<wslam::viz::VizGUI> gui_;
    std::unique_ptr<ResourceProvider> res_provider_;
    size_t current_idx_ = 0;
    size_t resource_count_ = 0;
    int frames_to_skip_ = 0;
    bool advance_frame_ = false;

    std::optional<std::string> drawResource(Resource res);
    void drawCorners(const Resource& res);
    void drawFeatures(const Resource& res);
    void drawMatches(const Resource& res);
    void drawLandmarks(const Resource& res);
    void initCallbacks();
    void initNextResourceCallback();
};
};  // namespace wslam::viz

namespace {

constexpr bool is_non_zero(std::uint8_t v) noexcept { return v != 0; }
constexpr bool is_non_zero(std::uint32_t v) noexcept { return v != 0; }
constexpr bool is_non_zero(std::int32_t v) noexcept { return v != 0; }
constexpr bool is_non_zero(const wslam::Corner& c) noexcept {
    return c.strength != 0;
}
constexpr bool is_non_zero(const wslam::Feature& f) noexcept {
    return f.strength != 0;
}

template <typename Container>
constexpr std::size_t count_non_zero(const Container& c) noexcept {
    return static_cast<std::size_t>(
        std::count_if(std::begin(c), std::end(c),
                      [](const auto& v) { return is_non_zero(v); }));
}

}  // namespace
   //
template <>
struct std::formatter<wslam::viz::CornerStyle> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static constexpr auto format(const wslam::viz::CornerStyle& s,
                                 std::format_context& ctx) {
        return std::format_to(ctx.out(),
                              "{{CornerStyle color:[{},{},{}] thickness:{} }}",
                              static_cast<unsigned>(s.color[0]),
                              static_cast<unsigned>(s.color[1]),
                              static_cast<unsigned>(s.color[2]),
                              static_cast<unsigned>(s.thickness));
    }
};

template <>
struct std::formatter<wslam::viz::FeatureStyle> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static constexpr auto format(const wslam::viz::FeatureStyle& s,
                                 std::format_context& ctx) {
        return std::format_to(
            ctx.out(),
            "{{FeatureStyle feature_color:[{},{},{}] "
            "bit_one_color:[{},{},{}] bit_zero_color:[{},{},{}] radius:{} }}",
            static_cast<unsigned>(s.feature_color[0]),
            static_cast<unsigned>(s.feature_color[1]),
            static_cast<unsigned>(s.feature_color[2]),
            static_cast<unsigned>(s.bit_one_color[0]),
            static_cast<unsigned>(s.bit_one_color[1]),
            static_cast<unsigned>(s.bit_one_color[2]),
            static_cast<unsigned>(s.bit_zero_color[0]),
            static_cast<unsigned>(s.bit_zero_color[1]),
            static_cast<unsigned>(s.bit_zero_color[2]), s.radius);
    }
};

template <>
struct std::formatter<wslam::viz::MatchStyle> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static constexpr auto format(const wslam::viz::MatchStyle& s,
                                 std::format_context& ctx) {
        return std::format_to(
            ctx.out(),
            "{{MatchStyle line_color:[{},{},{}] "
            "point_a_color:[{},{},{}] point_b_color:[{},{},{}] radius:{} }}",
            static_cast<unsigned>(s.line_color[0]),
            static_cast<unsigned>(s.line_color[1]),
            static_cast<unsigned>(s.line_color[2]),
            static_cast<unsigned>(s.point_a_color[0]),
            static_cast<unsigned>(s.point_a_color[1]),
            static_cast<unsigned>(s.point_a_color[2]),
            static_cast<unsigned>(s.point_b_color[0]),
            static_cast<unsigned>(s.point_b_color[1]),
            static_cast<unsigned>(s.point_b_color[2]), s.radius);
    }
};

template <>
struct std::formatter<wslam::viz::Resource> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static auto format(const wslam::viz::Resource& r,
                       std::format_context& ctx) {
        auto out
            = std::format_to(ctx.out(), "{{Resource title:\"{}\" texture:{}",
                             r.title, r.texture.has_value());
        if (r.corner_style) {
            out = std::format_to(out, " corner_style:{}", *r.corner_style);
        }
        if (r.corners) {
            out = std::format_to(out, " corners:{{ size:{} non-zero:{} }}",
                                 r.corners->size(), count_non_zero(*r.corners));
        }
        if (r.feature_style) {
            out = std::format_to(out, " feature_style:{}", *r.feature_style);
        }
        if (r.features) {
            out = std::format_to(out, " features:{{ size:{} non-zero:{} }}",
                                 r.features->size(),
                                 count_non_zero(*r.features));
        }

        out = std::format_to(out, " brief_tests:{}", !!r.brief_tests);

        if (r.match_style) {
            out = std::format_to(out, " match_style:{}", *r.match_style);
        }
        if (r.feature_matches) {
            out = std::format_to(out, " feature_matches:{{ size:{} }}",
                                 r.feature_matches->size());
        }

        return std::format_to(out, " }}");
    }
};
