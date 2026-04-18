#pragma once

#include <webgpu/webgpu_cpp.h>

#include <chrono>
#include <expected>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace wslam::compute {
struct FutureInfo {
    wgpu::Future future;
    std::string error_label;
};

class AwaiterTasks {
   public:
    struct Task {
        FutureInfo future_info = {};
        wgpu::FutureWaitInfo wait_info = {};
        bool is_completion_logged = false;
    };

    [[nodiscard]] std::optional<Task> at(size_t idx) const;
    [[nodiscard]] std::optional<AwaiterTasks::Task> getFuture(
        wgpu::Future future) const;

    [[nodiscard]] std::vector<wgpu::FutureWaitInfo>& getWaitInfos();

    void set(size_t idx, Task entry);
    void add(const FutureInfo& info,
             size_t idx = std::numeric_limits<size_t>::max());
    void add(Task task, size_t idx = std::numeric_limits<size_t>::max());
    size_t size();
    void clear();

    std::vector<wgpu::FutureWaitInfo>& get_wait_infos();

   private:
    std::vector<FutureInfo> future_infos_;
    std::vector<wgpu::FutureWaitInfo> wait_infos_;
    std::vector<bool> is_completion_logged_;
};

const int kDefaultTimeoutSeconds = 10;

class Awaiter {
   public:
    Awaiter(wgpu::Instance instance, wgpu::Device device)
        : instance_(std::move(instance)), device_(std::move(device)) {};

    // The factory is invoked immediately
    template <typename T>
        requires std::is_same_v<std::invoke_result_t<T>, wgpu::Future>
                 || std::is_same_v<std::invoke_result_t<T>, void>
    Awaiter& addCall(const T&& callback, std::string error_label,
                     bool catch_errors = true) {
        if constexpr (std::is_same_v<std::invoke_result_t<T>, wgpu::Future>) {
            return addCallFuture(std::move(callback), error_label,
                                 catch_errors);
        } else if constexpr (std::is_same_v<std::invoke_result_t<T>, void>) {
            return addCallVoid(std::move(callback), error_label, catch_errors);
        } else {
            static_assert(false, "Could not determine callback type");
        }
    }

    Awaiter& addFuture(wgpu::Future&& future, std::string error_label,
                       bool catch_errors = true);

    // Block until every enqueued future resolves or the timeout elapses.
    // Returns the first error encountered (label + detail), or
    // std::nullopt.
    [[nodiscard]] std::optional<std::string> executeAll(
        bool stop_on_error = true,
        std::chrono::nanoseconds timeout
        = std::chrono::seconds(kDefaultTimeoutSeconds));

   private:
    Awaiter& addCallFuture(std::function<wgpu::Future()>&& factory,
                           std::string error_label, bool catch_errors = true);

    Awaiter& addCallVoid(const std::function<void()>& callback,
                         std::string error_label, bool catch_errors = true);

    void addErrorHandlers(std::string error_label);

    [[nodiscard]] std::expected<bool, std::string> waitAny();

    wgpu::Instance instance_;
    wgpu::Device device_;
    AwaiterTasks tasks_;
    std::optional<std::string>
        pending_error_;  // set on immediate factory failure
};
}  // namespace wslam::compute
