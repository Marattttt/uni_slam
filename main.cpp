#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include "compute.hpp"
#include "euroc_provider.hpp"
#include "provider_base.hpp"
#include "wslam/common.hpp"
#include "wslam/feature_detect.hpp"

using namespace wslam;
using namespace std::chrono_literals;

int main_test() {
    compute::Compute comp;
    if (auto err = comp.preInitialize(compute::createPreInitializeOpts())) {
        spdlog::error("initializing: {}", err.value());
        std::terminate();
    }

    wslam::GpuSharedBindings shared(comp.getGPUPtr(), comp.getStorage());
    if (auto err = shared.initialize()) {
        spdlog::error("shared: {}", err.value());
        std::terminate();
    }

    std::unique_ptr<data::ProviderBase<2U>> euroc_data
        = std::make_unique<data::EurocProvider>(
            data::CreateEurocProviderOpts());

    auto euroc_generator = euroc_data->getReadings();

    auto data_provider = data::AdaptProvider<1UL, 2UL>(euroc_generator);

    comp.addStage(wslam::CreateFeatureDetectStage(comp, shared,
                                                  std::move(data_provider)));
    if (auto err = comp.initizalizeAllStages()) {
        spdlog::error("initializing stages: {}", err.value());
        std::terminate();
    }

    if (auto err = comp.execute()) {
        spdlog::error("executing: {}", err.value());
        std::terminate();
    }

    return 0;
}

int main() {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    return main_test();
}
