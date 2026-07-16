#pragma once

#include <webgpu/webgpu_cpp.h>

#include <filesystem>
#include <string>

#include "anybag.hpp"
#include "compute/gpu.hpp"

#ifndef WSLAM_FRAME_WIDTH
#define WSLAM_FRAME_WIDTH 752  // Euroc mav
#endif

#ifndef WSLAM_FRAME_HEIGHT
#define WSLAM_FRAME_HEIGHT 480  // Euroc mav
#endif

namespace wslam {

struct WslamConfig {
    bool enable_gui = false;

    // Stop after this many top-level pipeline executions. 0 means run until
    // the data provider is exhausted or a stage signals full stop.
    uint64_t max_iterations = 0;

    // Where the final map should be written when the pipeline exits. Must
    // end in `.ply`; the JSON metadata sidecar is derived by replacing the
    // extension. Empty means no export is performed.
    std::filesystem::path map_out_path;
};

struct LOD {
    uint8_t v;
    operator uint8_t() const { return v; }
};

namespace GPUConst {
constexpr uint32_t frame_width = WSLAM_FRAME_WIDTH;
constexpr uint32_t frame_height = WSLAM_FRAME_HEIGHT;
constexpr uint32_t pixel_size = sizeof(float);
constexpr uint32_t levels_of_detail = 6;
constexpr uint32_t min_uniform_buffer_alignment = 256;

// Assumed value
constexpr uint32_t max_features_per_lod = frame_width * frame_height / 250;

constexpr uint32_t featuesets_stored = 2;

constexpr double lod_scale_factor = 1.2;
};  // namespace GPUConst

[[nodiscard]] constexpr uint32_t CeilDiv(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

[[nodiscard]] constexpr auto AddPadding(auto num, size_t alignment) {
    alignment -= 1;
    return (num + alignment) & ~alignment;
}

template <typename T,
          std::size_t Alignment = GPUConst::min_uniform_buffer_alignment>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] std::vector<std::byte> AddPadding(std::span<const T> data) {
    constexpr std::size_t stride
        = (sizeof(T) + Alignment - 1) / Alignment * Alignment;

    std::vector<std::byte> result(stride * data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto src = std::as_bytes(data.subspan(i, 1));
        std::ranges::copy(src, result.begin() + i * stride);
    }
    return result;
}

namespace GPUBindingSize {
constexpr std::pair<size_t, size_t> getPyramidLayerDimensions(const LOD lod) {
    if (lod.v >= GPUConst::levels_of_detail) {
        throw std::out_of_range("LoD out of range");
    }

    double scale = 1.0;

    for (uint8_t i = 0; i < lod.v; ++i) {
        scale *= GPUConst::lod_scale_factor;
    }

    const auto width = static_cast<size_t>(GPUConst::frame_width / scale);
    const auto height = static_cast<size_t>(GPUConst::frame_height / scale);
    return {width, height};
}

constexpr size_t getPyramidLayerSize(const LOD lod) {
    const auto [width, height] = getPyramidLayerDimensions(lod);
    return width * height * GPUConst::pixel_size;
}

// Size (in bytes) of the initial frame
constexpr size_t source_frame = static_cast<size_t>(GPUConst::frame_width)
                                * GPUConst::frame_height * GPUConst::pixel_size;

// Maximum space a LoD pyramid can take up
constexpr size_t max_pyramid_size = std::invoke([]() consteval {
    size_t total = 0;
    for (uint8_t i = 0; i < GPUConst::levels_of_detail; ++i) {
        total += getPyramidLayerSize(LOD{i});
    }
    return total;
});
}  // namespace GPUBindingSize

namespace ResourceIdentifier {
constexpr std::string GetFrameName(std::pair<uint32_t, LOD> info) {
    return std::format("res:frame:{}:lod:{}", info.first, info.second.v);
}
constexpr std::string GetFrameName(uint32_t frame_idx) {
    return GetFrameName({frame_idx, {0}});
};
constexpr std::string GetTextureName(size_t lod) {
    return std::format("res:texture:lod:{}", lod);
}
constexpr std::string GetImuVecName() { return "res:imu:vec"; }
constexpr std::string GetVizResourceName() { return "res:viz"; }
constexpr std::string GetProcessedFrameName(uint32_t keyframes_ago,
                                            uint32_t lod) {
    return std::format("gen:frame:-{}:lod:{}", keyframes_ago, lod);
}
constexpr std::string GetFeatureSetName(size_t keyframes_ago) {
    return std::format("gen:feat:-{}", keyframes_ago);
}
constexpr std::string MatchedFeaturesName = "gen:feat:match";
// Consume-once marker (bool) set by the keyframe gate telling
// LoadDataCPUPass to advance the reference feature set: the *current*
// frame's features become FeatureSet(1) on the next iteration. Emitted
// when a keyframe is accepted (the reference must track the latest
// keyframe so matching/triangulation span exactly the graph edge) and on
// every pre-origin frame (frame-to-frame matching while bootstrapping).
// NOTE: constexpr std::string requires the literal to fit in SSO (15
// chars on libstdc++), same as every other key in this namespace.
constexpr std::string FeatureReferenceAdvanceName = "gen:feat:refadv";
constexpr std::string RansacResultName = "gen:feat:ransac";
constexpr std::string TriangulationResultName = "gen:tri";
constexpr std::string MapDeltaName = "gen:map:delta";
constexpr std::string MapSnapshotName = "gen:map:snap";
// Per-accepted-keyframe factor-graph delta produced by BuildFactorsPass and
// consumed by the iSAM update pass (graph + values + smart-factor bookkeeping).
constexpr std::string FactorBundleName = "gen:map:facbndl";

constexpr std::string GetCameraIntrinsicsName(uint32_t cam_idx) {
    return std::format("res:cam:intrinsics:{}", cam_idx);
}
constexpr std::string ImuParamsName = "res:imu:params";
// Latest frame timestamp (nanoseconds, monotonic) of the frame the sensor
// loader has just emitted. Mapping-stage passes read this to window the
// per-keyframe IMU interval.
constexpr std::string FrameTimestampNsName = "res:frame:ts_ns";
}  // namespace ResourceIdentifier

class FillPyramidPass;

class GpuSharedBindings {
   public:
    GpuSharedBindings(const std::shared_ptr<compute::GPU>& gpu, AnyBag& storage)
        : gpu_(gpu), storage(storage) {}

    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] constexpr wgpu::Texture getTexture(size_t lod) const {
        return textures_.at(lod);
    }
    [[nodiscard]] constexpr const AnyBag& getStorage() const { return storage; }
    [[nodiscard]] constexpr AnyBag& getStorage() { return storage; }

   private:
    friend FillPyramidPass;

    std::shared_ptr<compute::GPU> gpu_;
    AnyBag& storage;

    std::optional<std::string> initTextures();
    std::optional<std::string> initSrcTexture(compute::Awaiter& awaiter);
    void initTexture(compute::Awaiter& awaiter, uint32_t lod);

    std::array<wgpu::Texture, GPUConst::levels_of_detail> textures_;
};
};  // namespace wslam
