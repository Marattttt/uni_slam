#pragma once

#include "common.hpp"
#include "compute.hpp"
#include "provider_base.hpp"

namespace wslam {
namespace impl {
void CreateWslamPipelineImpl(
    compute::Compute& compute, GpuSharedBindings& shared,
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

};  // namespace wslam
