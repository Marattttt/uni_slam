#pragma once

#include "common.hpp"
#include "compute.hpp"
#include "provider_base.hpp"
#include "stage.hpp"

namespace wslam {
compute::Stage CreateFeatureDetectStage(
    compute::Compute& compute, GpuSharedBindings& shared_bindings,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    std::string feature_output_label);

};  // namespace wslam
