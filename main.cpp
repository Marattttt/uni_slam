#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <ranges>

#include "compute.hpp"
#include "euroc_provider.hpp"
#include "provider_base.hpp"
#include "wslam/presentable.hpp"

using namespace wslam;
using namespace std::chrono_literals;

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
int main_test() {
    compute::Compute comp;
    if (auto err = comp.initizalize(compute::createInitializeOpts(
            compute::DefinedWorkflow::HelloWGSL))) {
        spdlog::error("initializing: {}", err.value());
        std::terminate();
    }

    if (auto err = comp.execute()) {
        spdlog::error("executing: {}", err.value());
    }

    return 0;
}

int main_feature_detect() {
    compute::Compute compute;
    if (auto err = compute.initizalize(compute::createInitializeOpts())) {
        spdlog::error("{}", err.value());
        std::terminate();
    };

    data::EurocProvider euroc(data::CreateEurocProviderOpts());

    auto generator = euroc.getReadingsSynchronized();
    auto iter = generator.begin();
    std::vector<std::byte> frame_storage;

    ImageProvider img_provider
        = [&]() -> std::optional<std::span<const std::byte>> {
        if (iter == generator.end()) {
            return std::nullopt;
        }
        auto res = *iter;
        if (!res) {
            spdlog::error("[Image provider] could not get data reading: {}",
                          res.error());
            return std::nullopt;
        }

        auto frame = res.value().frame.front();
        std::span<std::uint8_t> frame_span{
            frame.pixels.data(), frame.pixels.data() + frame.pixels.size()};

        frame_storage = std::as_bytes(frame_span)
                        | std::ranges::to<std::vector<std::byte>>();

        return std::span(frame_storage);
    };

    wslam::Presentation pres{compute.getGPUPtr(), std::move(img_provider)};

    if (auto err = pres.initialize()) {
        spdlog::error("init presentation: {}", err.value());
        std::terminate();
    }

    while (true) {
        auto err = pres.execute();
        if (err) {
            spdlog::error("presentation execute: {}", err.value());
            break;
        }
    }

    return 0;
}

int main() { return main_feature_detect(); }
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
