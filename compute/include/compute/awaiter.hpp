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

// Don't mix runChecked and addFuture in the same Awaiter with a non-zero
// executeAll timeout: PopErrorScope is a WaitListEvent, OnSubmittedWorkDone
// / MapAsync are queue-serial, and Dawn rejects mixed-source timed waits
// (EventManager.cpp:211). For the submit-and-wait pattern, use
// GPU::submitAndWait.
class Awaiter {
   public:
    Awaiter(wgpu::Instance instance, wgpu::Device device)
        : instance_(std::move(instance)), device_(std::move(device)) {};

    // Run an API call inside PushErrorScope/PopErrorScope. The two pop
    // futures are tracked. The callback is invoked synchronously here.
    Awaiter& runChecked(const std::function<void()>& callback,
                        std::string label);

    // Call the factory, track the returned future. No error scope wrapping.
    // Use for OnSubmittedWorkDone, MapAsync, RequestAdapter,
    // CreatePipelineAsync.
    Awaiter& addFuture(const std::function<wgpu::Future()>& factory,
                       std::string label);

    // Block until every enqueued future resolves or the timeout elapses.
    // Returns the first error encountered (label + detail), or
    // std::nullopt.
    [[nodiscard]] std::optional<std::string> executeAll(
        bool stop_on_error = true,
        std::chrono::nanoseconds timeout
        = std::chrono::seconds(kDefaultTimeoutSeconds));

    [[nodiscard]] constexpr std::expected<void, std::string> execute(
        bool stop_on_error = true,
        std::chrono::nanoseconds timeout
        = std::chrono::seconds(kDefaultTimeoutSeconds)) {
        if (auto err = executeAll(stop_on_error, timeout)) {
            return std::unexpected(std::move(err).value());
        }
        return {};
    }

   private:
    void addErrorHandlers(std::string error_label);

    [[nodiscard]] std::expected<bool, std::string> waitAny(
        std::chrono::nanoseconds timeout);

    wgpu::Instance instance_;
    wgpu::Device device_;
    AwaiterTasks tasks_;
    std::optional<std::string>
        pending_error_;  // set on immediate factory failure
};
}  // namespace wslam::compute
