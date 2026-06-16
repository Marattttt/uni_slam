#include <spdlog/cfg/env.h>
#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <charconv>
#include <csignal>
#include <memory>
#include <span>
#include <string_view>

#include "compute.hpp"
#include "euroc_provider.hpp"
#include "provider_base.hpp"
#include "wslam/common.hpp"
#include "wslam/wslam.hpp"

using namespace wslam;
using namespace std::chrono_literals;

#define LOG_ID "[MAIN]"

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
                spdlog::warn(
                    LOG_ID " Ignoring malformed --max-iters value '{}'", value);
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

namespace {
volatile std::sig_atomic_t g_shutdown_requested = 0;

// Handle a shutdown signal
extern "C" void handleShutdownSignal(int signal) {
    (void)signal;

    g_shutdown_requested = 1;
    // A second interruption signal like a Ctrl+C kills the program
    // Helps if program hangs after the first shutdown request
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
}
}  // namespace

int main_test(WslamConfig config) {
    compute::Compute comp;
    if (auto err = comp.prepare(compute::createDefaultConfig())) {
        spdlog::critical(LOG_ID " initializing: {}", err.value());
        std::terminate();
    }

    wslam::GpuSharedBindings shared(comp.getGPUPtr(), comp.getStorage());
    if (auto err = shared.initialize()) {
        spdlog::critical(LOG_ID " shared: {}", err.value());
        std::terminate();
    }

    std::shared_ptr<data::ProviderBase<2U>> euroc_data
        = std::make_shared<data::EurocProvider>(
            data::CreateEurocProviderOpts());

    if (auto err
        = CreateWslamPipeline(comp, shared, std::move(euroc_data), config)) {
        spdlog::critical(LOG_ID " creating wslam pipeline: {}",
                         std::move(err).value());
        std::terminate();
    }

    if (auto err = comp.initizalize()) {
        spdlog::critical(LOG_ID " initializing stages: {}", err.value());
        std::terminate();
    }

    // Stop the loop cleanly on Ctrl+C (SIGINT) or kill (SIGTERM) so finalize()
    // still flushes the iSAM2 worker and exports the map. The handler re-arms
    // the OS defaults, so a second signal force-quits.
    std::signal(SIGINT, handleShutdownSignal);
    std::signal(SIGTERM, handleShutdownSignal);

    for (uint64_t i = 0;
         (config.max_iterations == 0 || i < config.max_iterations)
         && g_shutdown_requested == 0;
         ++i) {
        auto err = comp.execute();
        if (i % 100 == 0) {
            // Warn level so progress stays visible when info is disabled.
            spdlog::warn(LOG_ID " Handled frame {}", i);
        }

        if (err) {
            // If the interrupt landed mid-execute and surfaced as an error,
            // fall through to finalize rather than aborting the export.
            if (g_shutdown_requested != 0) {
                break;
            }
            spdlog::critical(LOG_ID " executing: {}", err.value());
            return 1;
        }
    }

    if (g_shutdown_requested != 0) {
        spdlog::info(LOG_ID
                     " Shutdown signal received; stopping loop and "
                     "finalizing (flush + export). Send the signal again to "
                     "force-quit.");
    } else if (config.max_iterations != 0) {
        spdlog::info(LOG_ID " Reached max-iters limit ({}); exiting",
                     config.max_iterations);
    }

    if (auto err = comp.finalize()) {
        spdlog::error(LOG_ID " Could not finalize. Errors: {}",
                      std::move(err).value());
    }

    return 0;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif
    // SPDLOG_LEVEL=info (etc.) overrides the compiled-in default at
    // runtime — e.g. to see the per-pass timing lines logged by Stage
    // without rebuilding.
    spdlog::cfg::load_env_levels();

    const WslamConfig config
        = parseArgs(std::span{argv + 1, static_cast<size_t>(argc - 1)});

    spdlog::info(
        LOG_ID " GUI: {}, max iterations: {}, map_out: {}", config.enable_gui,
        config.max_iterations == 0 ? std::string("unlimited")
                                   : std::to_string(config.max_iterations),
        config.map_out_path.empty() ? std::string("(none)")
                                    : config.map_out_path.string());
    return main_test(config);
}
