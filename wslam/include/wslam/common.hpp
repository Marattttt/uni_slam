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

    // Where compute runs: the Dawn adapter of either pipeline and, in v2, the
    // execution provider ONNX Runtime runs the model on. Has to match the
    // device compute::Compute was prepared with
    compute::ComputeDevice compute_device = compute::ComputeDevice::Gpu;

    // Stop after this many top-level pipeline executions. 0 means run until
    // the data provider is exhausted or a stage signals full stop.
    uint64_t max_iterations = 0;

    // Where the final map should be written when the pipeline exits. Must
    // end in `.ply`; the JSON metadata sidecar is derived by replacing the
    // extension. Empty means no export is performed.
    std::filesystem::path map_out_path;

    // Run the v2 pipeline (ONNX feature detect) instead of the default one
    bool use_pipeline_v2 = false;

    // Keyframe schedule (JSON, see keyframe_schedule.hpp). When set, the v2
    // pipeline only reads the listed frames, so every model run pairs two
    // consecutive keyframes; all IMU samples are still read. Empty means
    // every frame. v2 only
    std::filesystem::path keyframes_path;

    // Run the mapping stage after pose estimation. The model's reference frame
    // then tracks the keyframes mapping accepts; without mapping it advances
    // every frame. v2 only
    bool enable_mapping = true;

    // Record the front-end output (model matches, IMU, keyframe decisions) of
    // the run to this MessagePack file, for replaying the mapping stage without
    // the GPU (see frontend_recording.hpp). Empty means no recording. v2 with
    // mapping only
    std::filesystem::path frontend_recording_out;

    // Run loop closure (place recognition, PnP verification, a pose graph)
    // on a thread of its own, and export the loop-corrected map next to the
    // uncorrected one. v2 with mapping only
    bool enable_loop_closure = true;

    // Skip the model on frames whose motion since the last keyframe the IMU
    // predicts to be too small for the keyframe gate. v2 with mapping only,
    // and off with a keyframe schedule
    bool enable_frame_prefilter = true;
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
// The v2 pipeline uses it the same way: its PreservePyramidPass consumes it
// and makes the frame the model's reference image.
// NOTE: constexpr std::string requires the literal to fit in SSO (15
// chars on libstdc++), same as every other key in this namespace.
constexpr std::string FeatureReferenceAdvanceName = "gen:feat:refadv";
constexpr std::string RansacResultName = "gen:feat:ransac";
constexpr std::string TriangulationResultName = "gen:tri";
constexpr std::string MapDeltaName = "gen:map:delta";
constexpr std::string MapSnapshotName = "gen:map:snap";
// v2 with loop closure: the smoother's own (uncorrected, VIO) final
// snapshot. MapSnapshotName then holds the loop-corrected one
constexpr std::string MapSnapshotVioName = "gen:map:snapvio";
// v2 with loop closure and a global bundle adjustment: the pose graph's
// corrected map, before the adjustment. MapSnapshotName then holds the
// adjusted one
constexpr std::string MapSnapshotPgoName = "gen:map:snappgo";
// v2 loop closure outcome (LoopClosureReport), for the export
constexpr std::string LoopClosureReportName = "gen:lc:report";
// Per-accepted-keyframe factor-graph delta produced by BuildFactorsPass and
// consumed by the iSAM update pass (graph + values + smart-factor bookkeeping).
constexpr std::string FactorBundleName = "gen:map:facbndl";
// v2 mapping: keyframe accepted by the filter pass, before landmark
// association (v2::map::KeyframeCandidate)
constexpr std::string KeyframeCandidateName = "gen:map:kfcand";

constexpr std::string GetCameraIntrinsicsName(uint32_t cam_idx) {
    return std::format("res:cam:intrinsics:{}", cam_idx);
}
constexpr std::string ImuParamsName = "res:imu:params";
// Latest frame timestamp (nanoseconds, monotonic) of the frame the sensor
// loader has just emitted. Mapping-stage passes read this to window the
// per-keyframe IMU interval.
constexpr std::string FrameTimestampNsName = "res:frame:ts_ns";

// v2 pipeline (ONNX feature detect). Consume-once marker (bool) set by the
// detect pass on frames the model ran on, so readers of its outputs skip the
// first frame
constexpr std::string OrtKeypointsReadyName = "gen:ort:kp:rdy";
// std::vector<raco_lightglue::KeypointPosition>: the reference image's
// keypoints, then the current image's (raco_lightglue::PairKeypointsCount)
constexpr std::string OrtKeypointsName = "gen:ort:kp";
// std::shared_ptr<const raco_lightglue::FrameFeatures>, the keypoints and
// descriptors of the last frame the model ran on
constexpr std::string OrtFrameFeaturesName = "gen:ort:feat";
// v2::KeyframeMotionPrior of the last accepted keyframe, published by the v2
// mapping stage, erased when the next keyframe is accepted
constexpr std::string KeyframeMotionPriorName = "gen:kf:prior";
// v2::KeyframeMotionPrediction of a frame the prefilter let through
// (consume-once)
constexpr std::string KeyframeMotionPredictionName = "gen:kf:pred";
// Consume-once marker (bool) of a frame the prefilter skipped
constexpr std::string FrameSkippedName = "gen:frm:skip";
// std::vector<raco_lightglue::KeypointMatch>
constexpr std::string OrtMatchesName = "gen:ort:match";
// std::vector<float>, one score per OrtMatchesName row
constexpr std::string OrtMatchScoresName = "gen:ort:mscore";
// std::shared_ptr<v2::CpuModelInput>: host NCHW model input for
// --device=cpu, published by OrtCpuFeatureDetectPass::initialize() and
// filled by the v2 SensorLoaderPass. Absent on the GPU path
constexpr std::string CpuModelInputName = "res:mdl:in:cpu";
// v2 pose estimate stage, previous -> current frame of the model's pair.
// v2::KeypointMatches
constexpr std::string KeypointMatchesName = "gen:kp:match";
// v2::KeypointRansacResult
constexpr std::string KeypointRansacName = "gen:kp:ransac";
// v2::KeypointTriangulationResult
constexpr std::string KeypointTriangulationName = "gen:kp:tri";
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
