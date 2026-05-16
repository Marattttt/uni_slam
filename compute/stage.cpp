#include "stage.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>

#include "compute.hpp"
#include "pass.hpp"
#include "pass_hellowgsl.hpp"

using namespace wslam::compute;

Stage::Stage(std::string id, std::shared_ptr<GPU> gpu)
    : gpu_(std::move(gpu)), id_("[" + std::move(id) + "]") {}

std::optional<std::string> Stage::initialize() {
    spdlog::info("[Stage] Initializing stage {}", getId());

    for (std::unique_ptr<Pass>& pass : passes_) {
        auto err = pass->initialize();
        if (err) {
            return std::format("initializing pass {}: {}", pass->getId(),
                               err.value());
        }
    }

    return std::nullopt;
}

std::optional<std::string> Stage::execute() {
    spdlog::info("[Stage] Executing stage {}", getId());

    for (std::unique_ptr<Pass>& pass : passes_) {
        auto err = pass->execute();

        if (err.value_or("") == kStageStopExecution) {
            spdlog::info("[Stage] Received stop execution command from pass {}",
                         pass->getId());
            return {};
        }

        if (err.value_or("") == kComputeStopExecution) {
            spdlog::info(
                "[Stage] Received stop compute execution command from pass {}",
                pass->getId());
            return kComputeStopExecution;
        }

        if (err) {
            return std::format("executing pass {}: {}", pass->getId(),
                               err.value());
        }
    }

    return std::nullopt;
}

void Stage::add_pass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
}

void Stage::add_pass(std::vector<std::unique_ptr<Pass>> passes) {
    passes_.reserve(passes_.size() + passes.size());
    std::ranges::move(passes, std::back_inserter(passes_));
}

std::string Stage::getId() const { return id_; }

Stage wslam::compute::CreateHelloWgslStage(const std::shared_ptr<GPU>& gpu) {
    Stage stage{"Hello wgsl", gpu};
    stage.add_pass(std::make_unique<HelloWGSLPass>(gpu));
    return stage;
}
