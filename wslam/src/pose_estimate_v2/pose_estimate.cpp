#include "pose_estimate.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <utility>

#include "load_matches.hpp"
#include "ransac.hpp"
#include "triangulate.hpp"

using namespace wslam;

#define LOG_ID "[Pose Estimate v2 stage]"

std::unique_ptr<compute::Stage> v2::CreatePoseEstimateStage(
    compute::Compute& compute, std::shared_ptr<SharedBindings> bindings,
    const WslamConfig& config) {
    auto stage = std::make_unique<compute::Stage>("Pose Estimate v2",
                                                  &compute.getPerf());

    stage->add_pass(std::make_unique<LoadMatchesPass>(
        compute.getGPUPtr(), std::move(bindings), compute.getStorage()));

    stage->add_pass(std::make_unique<RansacPass>(compute.getStorage()));

    stage->add_pass(std::make_unique<TriangulatePass>(compute.getStorage()));

    if (config.enable_gui) {
        spdlog::warn(LOG_ID " The GUI is not ported to the v2 pipeline yet");
    }

    return stage;
}
