#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <charconv>
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
    constexpr std::string_view kIterFlag = "--max-iters=";
    WslamConfig config;
    for (std::string_view arg : args) {
        if (arg == "-gui") {
            config.enable_gui = true;
        } else if (arg.starts_with(kIterFlag)) {
            const auto value = arg.substr(kIterFlag.size());
            uint64_t parsed = 0;
            const auto [ptr, ec] = std::from_chars(
                value.data(), value.data() + value.size(), parsed);
            if (ec != std::errc{} || ptr != value.data() + value.size()) {
                spdlog::warn("Ignoring malformed --max-iters value '{}'", value);
                continue;
            }
            config.max_iterations = parsed;
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

    // Pull camera intrinsics out of the dataset and publish each camera into
    // the shared storage. Downstream geometric passes (triangulation, future
    // pose-only PnP) consume these without coupling to the provider type.
    auto sensor_params = euroc_data->getSensorParams();
    if (!sensor_params) {
        spdlog::error("getting sensor params: {}", sensor_params.error());
        std::terminate();
    }
    for (uint32_t i = 0; i < sensor_params->cams.size(); ++i) {
        const auto& cam = sensor_params->cams[i];
        spdlog::info("Camera {}: model={}, intrinsics=[{}], distortion=[{}]",
                     i, cam.camera_model, cam.intrinsics,
                     cam.distortion_coefficients);
        comp.getStorage().set(ResourceIdentifier::GetCameraIntrinsicsName(i),
                              cam);
    }

    auto euroc_generator = euroc_data->getReadings();

    auto data_provider = data::AdaptProvider<1UL, 2UL>(euroc_generator);

    CreateWslamPipeline(comp, shared, std::move(data_provider), config);

    if (auto err = comp.initizalizeAllStages()) {
        spdlog::error("initializing stages: {}", err.value());
        std::terminate();
    }

    for (uint64_t i = 0;
         config.max_iterations == 0 || i < config.max_iterations; ++i) {
        auto err = comp.execute();
        if (err) {
            spdlog::error("executing: {}", err.value());
            std::terminate();
        }
    }

    if (config.max_iterations != 0) {
        spdlog::info("Reached max-iters limit ({}); exiting cleanly",
                     config.max_iterations);
    }

    return 0;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    const WslamConfig config = parseArgs(std::span{argv + 1, static_cast<size_t>(argc - 1)});
    spdlog::info("GUI: {}, max iterations: {}", config.enable_gui,
                 config.max_iterations == 0 ? std::string("unlimited")
                                            : std::to_string(config.max_iterations));
    return main_test(config);
}
