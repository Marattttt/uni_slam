#include "presentable.hpp"

#include <spdlog/spdlog.h>

#include "common.hpp"

using namespace wslam;

#define LOG_ID "[Presentation stage]"

std::optional<std::string> Presentation::initialize() {
    spdlog::info(LOG_ID " initializing");

    const GPUConst constants{};

    auto constants_data = std::span<const std::byte, sizeof(GPUConst)>{
        reinterpret_cast<const std::byte*>(&constants), sizeof(constants)};

    if (auto err = shared_.initialize(*gpu_, constants_data)) {
        return "shared bindings: " + err.value();
    }

    if (auto err = fill_pyramid_.initialize()) {
        return "fill pyramid pass: " + err.value();
    }

    if (auto err = detect_corners_.initialize()) {
        return "detect corners pass: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> Presentation::execute() {
    if (auto err = fill_pyramid_.execute()) {
        return "fill pyramid: " + err.value();
    };

    if (auto err = detect_corners_.execute()) {
        return "detect corners: " + err.value();
    }

    return std::nullopt;
}
