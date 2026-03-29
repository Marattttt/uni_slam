#include "feature_detect.hpp"

#include "common.hpp"
#include "detect_corners.hpp"

using namespace wslam;

compute::Stage wslam::CreateFeatureDetectStage(
    const std::shared_ptr<compute::GPU>& gpu,
    GpuSharedBindings& shared_bindings) {
    compute::Stage stage("Feature detect", gpu);

    stage.add_pass(std::make_unique<PassDetectCorners>(
        gpu, shared_bindings,
        PassDetectCornersOpts{.frame_w = GPUConst::frame_width,
                              .frame_h = GPUConst::frame_height,
                              .pyr_levels = GPUConst::pyr_levels}));

    return stage;
}
