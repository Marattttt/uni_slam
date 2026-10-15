#include "config.hpp"

#include <cstdlib>
#include <string>

#include "map/utils.hpp"

using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;
namespace utils = wslam::map::utils;

namespace {
[[nodiscard]] const char* EnvOrNull(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}
}  // namespace

VprModelKind loop_closure::SelectModelKind(compute::ComputeDevice device,
                                           const LoopClosureConfig& config) {
    if (config.model_override) {
        return config.model_override.value();
    }
    return device == compute::ComputeDevice::Gpu ? VprModelKind::MegaLoc
                                                 : VprModelKind::MixVpr;
}

std::expected<LoopClosureConfig, std::string>
loop_closure::ReadLoopClosureConfigFromEnv(LoopClosureConfig defaults) {
    LoopClosureConfig config = std::move(defaults);

    if (const char* path = EnvOrNull("WSLAM_VPR_MEGALOC_MODEL")) {
        config.megaloc_model_path = path;
    }
    if (const char* path = EnvOrNull("WSLAM_VPR_MIXVPR_MODEL")) {
        config.mixvpr_model_path = path;
    }

    if (const char* model = EnvOrNull("WSLAM_VPR_MODEL")) {
        const std::string name{model};
        if (name == "megaloc") {
            config.model_override = VprModelKind::MegaLoc;
        } else if (name == "mixvpr") {
            config.model_override = VprModelKind::MixVpr;
        } else {
            return std::unexpected("WSLAM_VPR_MODEL is '" + name
                                   + "', expected megaloc or mixvpr");
        }
    }

    MatchAcceptance& acceptance = config.acceptance;
    acceptance.top_n = utils::EnvU32("WSLAM_LC_TOP_N", acceptance.top_n);
    acceptance.min_similarity = static_cast<float>(utils::EnvDouble(
        "WSLAM_LC_MIN_SIMILARITY", acceptance.min_similarity));
    acceptance.min_keyframe_gap = utils::EnvU32("WSLAM_LC_MIN_KEYFRAME_GAP",
                                                acceptance.min_keyframe_gap);
    acceptance.min_pnp_inliers
        = utils::EnvU32("WSLAM_LC_MIN_INLIERS", acceptance.min_pnp_inliers);

    config.node_gate.min_translation_m = utils::EnvDouble(
        "WSLAM_LC_NODE_MIN_TRANSLATION", config.node_gate.min_translation_m);
    config.node_gate.min_rotation_rad = utils::EnvDouble(
        "WSLAM_LC_NODE_MIN_ROTATION", config.node_gate.min_rotation_rad);

    config.intra_op_threads
        = utils::EnvInt("WSLAM_LC_THREADS", config.intra_op_threads);

    if (acceptance.top_n == 0) {
        return std::unexpected("WSLAM_LC_TOP_N must be at least 1");
    }

    return config;
}
