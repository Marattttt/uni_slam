#include "compute/stage.hpp"

#include <spdlog/spdlog.h>

#include <format>

#include "compute/compute.hpp"
#include "compute/pass.hpp"

using namespace wslam::compute;

Stage::Stage(std::string id, PerfRecorder* perf)
    : id_(std::move(id)), perf_(perf) {}

std::string Stage::getId() const { return id_; }

void Stage::add_pass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
}

std::optional<std::string> Stage::initialize() {
    spdlog::info("{} Initializing", getId());

    for (auto& pass : passes_) {
        const auto perfscope
            = perf_ != nullptr
                  ? std::optional{perf_->beginRecord(pass->getId())}
                  : std::nullopt;

        if (auto err = pass->initialize()) {
            return std::format("initializing pass {}: {}", pass->getId(),
                               err.value());
        }
    }

    return std::nullopt;
}

std::optional<std::string> Stage::execute() {
    spdlog::info("{} Executing", getId());

    for (auto& pass : passes_) {
        const auto perfscope
            = perf_ != nullptr
                  ? std::optional{perf_->beginRecord(pass->getId())}
                  : std::nullopt;

        auto err_opt = pass->execute();

        if (!err_opt) {
            continue;
        }

        auto& err = err_opt.value();
        if (err == kStageStopExecution) {
            spdlog::warn("{} stage stop execution requested by pass {}",
                         getId(), pass->getId());
            return {};
        }

        if (err == kComputeStopExecution) {
            return err;
        }

        return std::format("{}: {}", pass->getId(), std::move(err));
    }

    return std::nullopt;
}
