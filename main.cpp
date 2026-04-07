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

    wslam::GpuSharedBindings shared(comp.getGPUPtr());
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

int main() { return main_test(); }

// wslam::Viz create_viz(bool* is_next_requested);

// int viz_main() {
//     spdlog::set_level(spdlog::level::debug);
//
//     wslam::compute::Compute compute;
//     if (auto err = compute.initizalize(compute::createInitializeOpts(
//             compute::DefinedWorkflow::HelloWGSL))) {
//         std::println("Compute.initialize: {}", err.value());
//         return 1;
//     }
//
//     bool is_next_requested = false;
//
//     wslam::Viz viz = create_viz(&is_next_requested);
//     data::EurocProvider euroc{data::CreateEurocProviderOpts()};
//
//     for (const auto& reading : euroc.getReadingsSynchronized()) {
//         if (viz.windowShouldClose()) {
//             break;
//         }
//
//         if (!reading) {
//             spdlog::error("getting reading: {}", reading.error());
//             viz.closeWindow();
//             break;
//         }
//
//         const auto& pixels = reading->frame[0].pixels;
//
//         const VizTexture texture{
//             .width = reading->frame.front().width,
//             .height = reading->frame.front().height,
//             .channels = 1,
//             .data = {pixels.data(), pixels.data() + pixels.size()}};
//
//         is_next_requested = false;
//
//         while (!viz.windowShouldClose() && !is_next_requested) {
//             viz.startFrame();
//             viz.drawTexture(texture);
//             viz.endFrame();
//         }
//     }
//
//     return 0;
// }
//
//
// wslam::Viz create_viz(bool* is_next_requested) {
//     auto viz_err = wslam::Viz::initialize({});
//     if (!viz_err.has_value()) {
//         std::println(stderr, "Could not create window. reason: {}",
//                      viz_err.error());
//     }
//
//     viz_err.value().addRequestNextCallback([&is_next_requested]() {
//         static std::chrono::time_point<std::chrono::steady_clock>
//         last_call{}; const auto delay = 50ms; const auto now =
//         std::chrono::steady_clock::now();
//
//         if (now - last_call > delay) {
//             last_call = now;
//             *is_next_requested = true;
//         }
//     });
//
//     return std::move(viz_err.value());
// }
