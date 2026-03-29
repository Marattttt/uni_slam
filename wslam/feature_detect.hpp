#pragma once

#include "common.hpp"
#include "stage.hpp"

namespace wslam {
compute::Stage CreateFeatureDetectStage(
    const std::shared_ptr<compute::GPU>& gpu,
    GpuSharedBindings& shared_bindings);

};  // namespace wslam
