#include "stage.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <ranges>
#include <string_view>
#include <variant>

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

    const auto device = gpu_->getDevice();
    const auto queue = device.GetQueue();

    const auto encoder = std::invoke([&] {
        const std::string label
            = std::format("command encoder for passes {{ {} }}", pass_names);
        const wgpu::CommandEncoderDescriptor desc{.label = label.c_str()};
        return device.CreateCommandEncoder(&desc);
    });

    std::vector<wgpu::CommandBuffer> commands;
    commands.reserve(batch.size());

    for (auto* pass : batch) {
        if (auto buffer = pass->prepareExecute(encoder)) {
            commands.emplace_back(std::move(buffer).value());
        } else {
            return std::format("preparing pass {}: {}", pass->getId(),
                               std::move(buffer).error());
        }
    }

    queue.Submit(commands.size(), commands.data());

    struct UserData {
        std::string error;
        std::string stage_id;
    };

    UserData data{
        .error = "",
        .stage_id = getId(),
    };

    auto wait = [&] {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView msg,
               UserData* data) {
                spdlog::info(
                    "[{}] finished batched gpu work. status:{}, msg:'{}'",
                    data->stage_id, static_cast<int>(status),
                    static_cast<std::string>(msg));
            },
            &data);
    };

    auto err
        = gpu_->getAwaiter()
              .addCall(std::move(wait), "execute gpu passes")
              .executeAll(false)
              .transform([&](auto&& err) {
                  return std::format("executing {{ {} }}: {}", pass_names, err);
              });

    if (data.error != "") {
        spdlog::error("[Stage] Error executing gpu passes {{ {} }}: '{}'",
                      pass_names, data.error);
    }

    return err;
}

void Stage::add_pass(PassPtr pass) { passes_.emplace_back(std::move(pass)); }

void Stage::add_pass(std::vector<std::unique_ptr<Pass>> passes) {
    passes_.reserve(passes_.size() + passes.size());
    std::ranges::move(passes, std::back_inserter(passes_));
}

std::string Stage::getId() const { return id_; }
