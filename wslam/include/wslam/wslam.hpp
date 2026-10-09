#pragma once

#include "common.hpp"
#include "compute/compute.hpp"
#include "data/provider_base.hpp"

namespace wslam {
namespace impl {
void CreateWslamPipelineImpl(
    compute::Compute& compute, GpuSharedBindings& shared,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const WslamConfig& config);

[[nodiscard]] std::optional<std::string> CreateWslamPipelineV2Impl(
    compute::Compute& compute,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    const WslamConfig& config);

template <size_t CamCnt>
[[nodiscard]] std::optional<std::string> SetSensorIntrinsics(
    compute::Compute& comp, data::ProviderBase<CamCnt>& provider) {
    auto sensor_params = provider.getSensorParams();
    if (!sensor_params) {
        return "getting sensor params: " + std::move(sensor_params).error();
    }

    spdlog::info("Sensor params: {}", sensor_params.value());

    for (uint32_t i = 0; i < sensor_params->cams.size(); ++i) {
        const auto& cam = sensor_params->cams[i];
        comp.getStorage().set(ResourceIdentifier::GetCameraIntrinsicsName(i),
                              cam);
    }

    comp.getStorage().set(ResourceIdentifier::ImuParamsName,
                          sensor_params->imu);

    return {};
}
};  // namespace impl

template <size_t CamCnt>
[[nodiscard]] std::optional<std::string> CreateWslamPipeline(
    compute::Compute& compute, GpuSharedBindings& shared,
    data::ProviderBase<CamCnt>& provider, const WslamConfig& config) {
    if (auto err = impl::SetSensorIntrinsics(compute, provider)) {
        return "getting sensor intrinsics: " + std::move(err).value();
    }

    impl::CreateWslamPipelineImpl(compute, shared,
                                  data::AdaptProvider<1U>(provider), config);

    return {};
}

// Pipeline built around the RaCo-ALIKED-LightGlue ONNX model. For now it only
// has the feature detect stage
template <size_t CamCnt>
[[nodiscard]] std::optional<std::string> CreateWslamPipelineV2(
    compute::Compute& compute, data::ProviderBase<CamCnt>& provider,
    const WslamConfig& config) {
    if (auto err = impl::SetSensorIntrinsics(compute, provider)) {
        return "getting sensor intrinsics: " + std::move(err).value();
    }

    if (auto err = impl::CreateWslamPipelineV2Impl(
            compute, data::AdaptProvider<1U>(provider), config)) {
        return "creating v2 pipeline: " + std::move(err).value();
    }

    return {};
}

};  // namespace wslam
