#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "compute/gpu.hpp"

// Configuration of the loop closure thread, shared by both place recognition
// models and handed over when the thread is created
namespace wslam::v2::mapping::loop_closure {

// Place recognition (global descriptor) models. MegaLoc runs when compute is
// on the GPU, MixVPR when it is on the CPU, unless the config overrides it
enum class VprModelKind : uint8_t { MegaLoc, MixVpr };

[[nodiscard]] constexpr std::string_view ToString(VprModelKind kind) {
    switch (kind) {
        case VprModelKind::MegaLoc:
            return "MegaLoc";
        case VprModelKind::MixVpr:
            return "MixVPR";
    }
    return "unknown";
}

// When a retrieved keyframe counts as a loop closure of the query keyframe.
// Retrieval, matching and PnP gates, applied in that order
struct MatchAcceptance {
    // Candidates FAISS returns per query
    uint32_t top_n = 3;
    // Cosine similarity of the two global descriptors
    float min_similarity = 0.5F;
    // A candidate must be this many keyframes and seconds older than the
    // query, so the recent trajectory (which odometry already links) is not
    // "recognised"
    uint32_t min_keyframe_gap = 30;
    double min_time_gap_s = 10.0;

    // RaCo-LightGlue matches between the two images, at or above the score
    float matcher_min_score = 0.5F;
    uint32_t min_matches = 40;

    // A candidate keypoint observes a landmark when it lies this close to a
    // keypoint the keyframe tracked one with
    double landmark_pixel_tolerance_px = 1.0;
    // 2D (query) - 3D (candidate landmark) pairs PnP needs
    uint32_t min_correspondences = 20;

    // PnP RANSAC
    double pnp_reprojection_px = 3.0;
    double pnp_confidence = 0.999;
    uint32_t pnp_max_iterations = 1000;
    uint32_t min_pnp_inliers = 20;
    double min_inlier_ratio = 0.3;

    // The PnP relative pose may disagree with odometry's by at most this much
    // (the drift accumulated since the candidate); larger is a false match
    double max_correction_translation_m = 2.0;
    double max_correction_rotation_rad = 0.35;
};

// A keyframe becomes a pose graph node (and a place in the database) only
// when it moved this far or turned this much from the last node; otherwise
// it is attached to that node
struct NodeGate {
    double min_translation_m = 0.10;
    double min_rotation_rad = 0.087;
};

// Noise of the pose-only graph
struct PoseGraphNoise {
    double prior_sigma = 1e-4;
    // Per odometry edge: a little above the v2 smoother's measured relative
    // error on V1_01_easy (0.18 deg and 1.1 cm per keyframe)
    double odometry_rotation_sigma_rad = 0.005;
    double odometry_translation_sigma_m = 0.02;
    // PnP on landmarks a few metres away
    double loop_rotation_sigma_rad = 0.005;
    double loop_translation_sigma_m = 0.03;
    // Cauchy kernel width on loop factors, in sigmas; 0 = no kernel. Off by
    // default: a true loop closes drift many sigmas wide, which a kernel
    // would mostly ignore, and verification (PnP inliers plus the
    // consistency bound) already rejects false loops
    double loop_robust_k = 0.0;
    // Roll and pitch: gravity makes them observable to the VIO, so the graph
    // holds them close to its estimate
    double attitude_sigma_rad = 0.01;
    // iSAM2 updates after a loop factor was added, to converge on it
    uint32_t extra_updates_after_loop = 3;
};

struct LoopClosureConfig {
    // Unset picks the model from the compute device
    std::optional<VprModelKind> model_override;

    std::filesystem::path megaloc_model_path;
    std::filesystem::path mixvpr_model_path;

    // RaCo-LightGlue pipeline model for the geometric verification (the
    // front end's model) and its keypoints per image
    std::filesystem::path matcher_model_path;
    int64_t matcher_keypoint_count = 0;
    // The matcher shares the GPU only on request: next to the front end and
    // MegaLoc it can exhaust a small integrated GPU's memory
    bool matcher_on_gpu = false;

    // Intra-op threads of each loop closure session on ONNX Runtime's CPU EP,
    // so the thread doesn't compete for every core with the front end. The
    // sessions never spin-wait. With --device=cpu the front end's session
    // gets the physical cores minus these (see CreateWslamPipelineV2Impl)
    int intra_op_threads = 4;

    MatchAcceptance acceptance;
    NodeGate node_gate;
    PoseGraphNoise noise;

    // On flush, process the keyframes still queued (true) or drop them
    bool flush_processes_backlog = true;
};

// The model the config and the compute device select
[[nodiscard]] VprModelKind SelectModelKind(compute::ComputeDevice device,
                                           const LoopClosureConfig& config);

// Reads the model paths and the overridable knobs from the environment:
//   WSLAM_VPR_MEGALOC_MODEL, WSLAM_VPR_MIXVPR_MODEL  model files
//   WSLAM_VPR_MODEL=megaloc|mixvpr                    model override
//   WSLAM_LC_TOP_N, WSLAM_LC_MIN_SIMILARITY, WSLAM_LC_MIN_KEYFRAME_GAP,
//   WSLAM_LC_MIN_INLIERS, WSLAM_LC_NODE_MIN_TRANSLATION,
//   WSLAM_LC_NODE_MIN_ROTATION, WSLAM_LC_THREADS     acceptance / runtime
// matcher_model_path and matcher_keypoint_count are the caller's
[[nodiscard]] std::expected<LoopClosureConfig, std::string>
ReadLoopClosureConfigFromEnv(LoopClosureConfig defaults = {});

}  // namespace wslam::v2::mapping::loop_closure
