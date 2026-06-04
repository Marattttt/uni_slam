#include <spdlog/cfg/env.h>
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
#include "wslam/export.hpp"
#include "wslam/wslam.hpp"

using namespace wslam;
using namespace std::chrono_literals;

WslamConfig parseArgs(std::span<char*> args) {
    constexpr std::string_view kIterFlag = "--max-iters=";
    constexpr std::string_view kMapOutFlag = "--map-out=";
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
                spdlog::warn("Ignoring malformed --max-iters value '{}'",
                             value);
                continue;
            }
            config.max_iterations = parsed;
        } else if (arg.starts_with(kMapOutFlag)) {
            config.map_out_path = std::filesystem::path(
                std::string(arg.substr(kMapOutFlag.size())));
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
        spdlog::info("Camera {}: model={}, intrinsics=[{}], distortion=[{}]", i,
                     cam.camera_model, cam.intrinsics,
                     cam.distortion_coefficients);
        comp.getStorage().set(ResourceIdentifier::GetCameraIntrinsicsName(i),
                              cam);
    }
    // IMU calibration (T_BS, noise densities, rate). Consumed by the
    // factor builder to configure preintegration; by the keyframe gate to
    // express gravity in the camera frame.
    spdlog::info(
        "IMU: rate={} Hz, gyro_nd={:.3e} rad/s/sqrt(Hz), "
        "accel_nd={:.3e} m/s^2/sqrt(Hz)",
        sensor_params->imu.rate_hz, sensor_params->imu.gyroscope_noise_density,
        sensor_params->imu.accelerometer_noise_density);
    comp.getStorage().set(ResourceIdentifier::ImuParamsName,
                          sensor_params->imu);

    auto euroc_generator = euroc_data->getReadings();

    auto data_provider = data::AdaptProvider<1UL, 2UL>(euroc_generator);

    auto handles
        = CreateWslamPipeline(comp, shared, std::move(data_provider), config);

    if (auto err = comp.initizalizeAllStages()) {
        spdlog::error("initializing stages: {}", err.value());
        std::terminate();
    }

    for (uint64_t i = 0;
         config.max_iterations == 0 || i < config.max_iterations; ++i) {
        auto err = comp.execute();
        if (err) {
            spdlog::error("executing: {}", err.value());
            return 1;
        }
    }

    if (config.max_iterations != 0) {
        spdlog::info("Reached max-iters limit ({}); exiting cleanly",
                     config.max_iterations);
    }

    // The iSAM2 update pass runs on a worker thread, lagging the main loop
    // by one frame. Drain any pending optimisation so the snapshot under
    // MapSnapshotName reflects every submitted keyframe before we read it
    // for export.
    if (handles.flush_async) {
        if (auto err = handles.flush_async()) {
            spdlog::error("flushing iSAM2 worker: {}", err.value());
            return 1;
        }
    }

    if (!config.map_out_path.empty()) {
        if (auto err = ExportMap(comp.getStorage(),
                                 ExportOpts{.map_path = config.map_out_path})) {
            spdlog::error("exporting map: {}", err.value());
            return 1;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::warn);
#endif
    // SPDLOG_LEVEL=info (etc.) overrides the compiled-in default at
    // runtime — e.g. to see the per-pass timing lines logged by Stage
    // without rebuilding.
    spdlog::cfg::load_env_levels();

    const WslamConfig config
        = parseArgs(std::span{argv + 1, static_cast<size_t>(argc - 1)});
    spdlog::info("GUI: {}, max iterations: {}, map_out: {}", config.enable_gui,
                 config.max_iterations == 0
                     ? std::string("unlimited")
                     : std::to_string(config.max_iterations),
                 config.map_out_path.empty() ? std::string("(none)")
                                             : config.map_out_path.string());
    return main_test(config);
}
