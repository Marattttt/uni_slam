#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <span>
#include <string_view>

#include "compute.hpp"
#include "euroc_provider.hpp"
#include "provider_base.hpp"
#include "wslam/common.hpp"
#include "wslam/wslam.hpp"

using namespace wslam;
using namespace std::chrono_literals;

WslamConfig parseArgs(std::span<char*> args) {
    WslamConfig config;
    for (std::string_view arg : args) {
        if (arg == "-gui") {
            config.enable_gui = true;
        }
    }
    return config;
}

int main_test(WslamConfig config) {
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

    CreateWslamPipeline(comp, shared, std::move(data_provider), config);

    if (auto err = comp.initizalizeAllStages()) {
        spdlog::error("initializing stages: {}", err.value());
        std::terminate();
    }

    while (true) {
        auto err = comp.execute();
        if (err) {
            spdlog::error("executing: {}", err.value());
            std::terminate();
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    const WslamConfig config = parseArgs(std::span{argv + 1, static_cast<size_t>(argc - 1)});
    spdlog::info("GUI: {}", config.enable_gui);
    return main_test(config);
}
