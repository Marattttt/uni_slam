#pragma once

#include <memory>

#include "common.hpp"
#include "compute/compute.hpp"
#include "data/provider_base.hpp"
#include "compute/stage.hpp"

namespace wslam {
std::unique_ptr<compute::Stage> CreateFeatureDetectStage(
    compute::Compute& compute, GpuSharedBindings& shared_bindings,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const std::string& feature_output_label, WslamConfig config = {});

};  // namespace wslam
