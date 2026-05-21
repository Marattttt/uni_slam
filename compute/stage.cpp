#include "stage.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <ranges>
#include <string_view>
#include <variant>

#include "compute.hpp"
#include "pass.hpp"

using namespace wslam::compute;

Stage::Stage(std::string id, std::shared_ptr<GPU> gpu)
    : gpu_(std::move(gpu)), id_("[" + std::move(id) + "]") {}

std::optional<std::string> Stage::initialize() {
    spdlog::info("[Stage] Initializing stage {}", getId());

    for (auto& pass : passes_) {
        std::optional<std::string> err
            = std::visit([](auto&& p) { return p->initialize(); }, pass);

        const std::string id
            = std::visit([](auto&& p) { return p->getId(); }, pass);

        if (err) {
            return std::format("initializing pass {}: {}", id, err.value());
        }
    }

    return std::nullopt;
}

namespace {
template <typename T>
    requires(std::is_same_v<T, Pass> || std::is_same_v<T, GPUPass>)
std::string CollectPassNames(std::span<T*> passes) {
    return passes | std::views::transform([](auto&& p) { return p->getId(); })
           | std::views::join_with(std::string_view(", "))
           | std::ranges::to<std::string>();
}
};  // namespace

std::optional<std::string> Stage::execute() {
    spdlog::info("[Stage] Executing stage {}", getId());

    // Batch gpu passes to execute in a single queue submission and await
    std::vector<GPUPass*> gpu_batch;

    // Only two types of passes are handled, overloaded type for visining an
    // std::variant is not used
    static_assert(std::variant_size_v<PassPtr> == 2);

    for (auto& pass : passes_) {
        if (auto* gpu_pass = std::get_if<std::unique_ptr<GPUPass>>(&pass)) {
            gpu_batch.emplace_back(gpu_pass->get());
            continue;
        }

        auto& cpu_pass = std::get<std::unique_ptr<Pass>>(pass);

        if (auto err = executeGPUBatch(std::span(gpu_batch))) {
            return err;
        }
        gpu_batch.clear();

        if (auto err = cpu_pass->execute()) {
            if (err.value() == kStageStopExecution) {
                spdlog::info("[Stage] {} stage-stop from {}", getId(),
                             cpu_pass->getId());
                return std::nullopt;
            }
            if (err.value() == kComputeStopExecution) {
                return err;
            }
            return std::format("pass {}: {}", cpu_pass->getId(),
                               std::move(err).value());
        }
    }

    if (auto err = executeGPUBatch(std::span(gpu_batch))) {
        return err;
    }

    return std::nullopt;
}

std::optional<std::string> Stage::executeGPUBatch(std::span<GPUPass*> batch) {
    if (batch.size() == 0) {
        return std::nullopt;
    }

    const std::string pass_names = CollectPassNames(batch);

    const auto encoder = std::invoke([&] {
        const std::string label
            = std::format("command encoder for passes {{ {} }}", pass_names);
        const wgpu::CommandEncoderDescriptor desc{.label = label.c_str()};
        return gpu_->getDevice().CreateCommandEncoder(&desc);
    });

    for (auto* pass : batch) {
        if (auto err = pass->prepareExecute(encoder)) {
            return std::format("preparing pass {}: {}", pass->getId(),
                               std::move(err).value());
        }
    }

    const auto commands = encoder.Finish();

    return gpu_->submitAndWait(
        commands, std::format("stage passes {{ {} }}", pass_names));
}

void Stage::add_pass(PassPtr pass) { passes_.emplace_back(std::move(pass)); }

void Stage::add_pass(std::vector<std::unique_ptr<Pass>> passes) {
    passes_.reserve(passes_.size() + passes.size());
    std::ranges::move(passes, std::back_inserter(passes_));
}

std::string Stage::getId() const { return id_; }
