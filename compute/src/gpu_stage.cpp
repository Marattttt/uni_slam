#include "compute/gpu_stage.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <format>
#include <functional>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <variant>

#include "compute/compute.hpp"
#include "compute/pass.hpp"

using namespace wslam::compute;

GpuStage::GpuStage(std::string id, std::shared_ptr<GPU> gpu, PerfRecorder* perf)
    : Stage(std::move(id), perf), gpu_(std::move(gpu)) {}

void GpuStage::add_pass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
}

void GpuStage::add_pass(std::unique_ptr<GPUPass> pass) {
    passes_.emplace_back(std::move(pass));
}

namespace {
[[nodiscard]] std::string GetPassId(const GpuStage::PassPtr& pass) {
    return std::visit([](auto& p) { return p->getId(); }, pass);
}
};  // namespace

std::optional<std::string> GpuStage::initialize() {
    spdlog::info("{} Initializing", getId());

    for (auto& pass : passes_) {
        const auto perfscope
            = perf_ != nullptr
                  ? std::optional{perf_->beginRecord(GetPassId(pass))}
                  : std::nullopt;

        std::optional<std::string> err
            = std::visit([](auto&& p) { return p->initialize(); }, pass);

        if (err) {
            return std::format("initializing pass {}: {}", GetPassId(pass),
                               err.value());
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

std::optional<std::string> GpuStage::executeGPUBatch(
    std::span<GPUPass*> batch) {
    if (batch.empty()) {
        return std::nullopt;
    }

    const std::string pass_names = CollectPassNames(batch);

    const auto perfscope = perf_ != nullptr
                               ? std::optional{perf_->beginRecord(std::format(
                                     "GPU Batch {{ {} }}", pass_names))}
                               : std::nullopt;

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

std::optional<std::string> GpuStage::runBatched() {
    // Batch consecutive GPU passes into a single queue submission
    std::vector<GPUPass*> gpu_batch;

    static_assert(std::variant_size_v<PassPtr> == 2,
                  "Only CPU and GPU passes are handled");

    for (auto& pass : passes_) {
        if (auto* gpu_pass = std::get_if<std::unique_ptr<GPUPass>>(&pass)) {
            gpu_batch.emplace_back(gpu_pass->get());
            continue;
        }

        if (auto err = executeGPUBatch(std::span(gpu_batch))) {
            return err;
        }
        gpu_batch.clear();

        auto& cpu_pass = std::get<std::unique_ptr<Pass>>(pass);

        auto err = cpu_pass->execute();

        if (!err) {
            continue;
        }

        if (err.value() == kStageStopExecution) {
            spdlog::warn("{} stage stop execution requested by pass {}",
                         getId(), GetPassId(pass));
            return std::nullopt;
        }

        if (err.value() == kComputeStopExecution) {
            return err;
        }

        return std::format("{}: {}", GetPassId(pass), err.value());
    }

    return executeGPUBatch(std::span(gpu_batch));
}

std::optional<std::string> GpuStage::execute() {
    spdlog::info("[Stage] Executing stage {}", getId());

#ifndef NPERF
    if (perf_ != nullptr) {
        const auto perfscope = perf_->beginRecord("exec " + getId());
        return runBatched();
    }
#endif

    return runBatched();
}
