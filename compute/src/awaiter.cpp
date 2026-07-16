#include "compute/awaiter.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <chrono>
#include <ranges>
#include <utility>

using namespace wslam::compute;
using namespace std::chrono_literals;

#define LOG_ID "[Awaiter]"

std::optional<AwaiterTasks::Task> AwaiterTasks::at(size_t idx) const {
    if (idx >= future_infos_.size() || idx >= wait_infos_.size()
        || idx >= is_completion_logged_.size()) {
        return std::nullopt;
    }

    return Task{
        .future_info = future_infos_[idx],
        .wait_info = wait_infos_[idx],
        .is_completion_logged = is_completion_logged_[idx],
    };
}

std::optional<AwaiterTasks::Task> AwaiterTasks::getFuture(
    wgpu::Future future) const {
    size_t idx = 0;

    for (size_t i = 0; i < future_infos_.size(); i++) {
        if (future_infos_[i].future.id == future.id) {
            idx = i;
            break;
        }
    }

    return at(idx);
}

std::vector<wgpu::FutureWaitInfo>& AwaiterTasks::getWaitInfos() {
    return wait_infos_;
}

std::vector<wgpu::FutureWaitInfo>& AwaiterTasks::get_wait_infos() {
    return wait_infos_;
}

void AwaiterTasks::set(size_t idx, Task entry) {
    if (idx >= future_infos_.size()) {
        throw std::out_of_range("AwaiterTasks::set index out of range");
    }

    future_infos_[idx] = std::move(entry.future_info);
    wait_infos_[idx] = entry.wait_info;
    is_completion_logged_[idx] = entry.is_completion_logged;
}

void AwaiterTasks::add(const FutureInfo& info, size_t idx) {
    add(
        {
            .future_info = info,
            .wait_info = wgpu::FutureWaitInfo{.future = info.future},
            .is_completion_logged = false,
        },
        idx);
}

void AwaiterTasks::add(Task task, size_t idx) {
    if (idx == std::numeric_limits<size_t>::max()) {
        future_infos_.push_back(std::move(task.future_info));
        wait_infos_.push_back(task.wait_info);
        is_completion_logged_.push_back(task.is_completion_logged);
        return;
    }

    if (idx > future_infos_.size()) {
        throw std::out_of_range("AwaiterTasks::add index out of range");
    }

    future_infos_.insert(future_infos_.begin() + static_cast<long>(idx),
                         std::move(task.future_info));
    wait_infos_.insert(wait_infos_.begin() + static_cast<long>(idx),
                       task.wait_info);
    is_completion_logged_.insert(
        is_completion_logged_.begin() + static_cast<long>(idx),
        task.is_completion_logged);
}

size_t AwaiterTasks::size() { return future_infos_.size(); }

void AwaiterTasks::clear() {
    future_infos_.clear();
    wait_infos_.clear();
    is_completion_logged_.clear();
}

void Awaiter::addErrorHandlers(std::string error_label) {
    wgpu::Future validation = device_.PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [this, error_label](wgpu::PopErrorScopeStatus status,
                            wgpu::ErrorType type, wgpu::StringView message) {
            if (status != wgpu::PopErrorScopeStatus::Success) {
                return;
            }
            if (type == wgpu::ErrorType::NoError) {
                return;
            }
            pending_error_ = pending_error_.value_or("")
                             + std::format("[{} (validation): {}]", error_label,
                                           std::string_view(message));
        });

    tasks_.add(FutureInfo{
        .future = validation,
        .error_label = std::format("{}, validation errscope", error_label)});

    wgpu::Future out_of_mem = device_.PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [this, error_label](wgpu::PopErrorScopeStatus status,
                            wgpu::ErrorType type, wgpu::StringView message) {
            if (status != wgpu::PopErrorScopeStatus::Success) {
                return;
            }
            if (type == wgpu::ErrorType::NoError) {
                return;
            }
            pending_error_
                = pending_error_.value_or("")
                  + std::format("[{} (out of memory): {}]", error_label,
                                std::string_view(message));
        });

    tasks_.add(FutureInfo{
        .future = out_of_mem,
        .error_label = std::format("{}, out of memory errscope", error_label)});
}

Awaiter& Awaiter::runChecked(const std::function<void()>& callback,
                             std::string label) {
    if (pending_error_) {
        return *this;
    }

    assert(label.length() > 0 && "label must be non-empty");

    device_.PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
    device_.PushErrorScope(wgpu::ErrorFilter::Validation);

#ifdef LOG_AWAITER_CALLS
    spdlog::debug(LOG_ID " Started runChecked call {}", label);
#endif

    callback();

    addErrorHandlers(std::move(label));

    return *this;
}

Awaiter& Awaiter::addFuture(const std::function<wgpu::Future()>& factory,
                            std::string label) {
    if (pending_error_) {
        return *this;
    }

    assert(label.length() > 0 && "label must be non-empty");

#ifdef LOG_AWAITER_CALLS
    spdlog::debug(LOG_ID " Started addFuture call {}", label);
#endif

    tasks_.add(FutureInfo{
        .future = factory(),
        .error_label = std::move(label),
    });

    return *this;
}

std::optional<std::string> Awaiter::executeAll(
    bool stop_on_error, std::chrono::nanoseconds timeout) {
    if (pending_error_) {
        tasks_.clear();
        return std::exchange(pending_error_, std::nullopt);
    }

    if (tasks_.size() == 0) {
        return std::nullopt;
    }

    const auto start_time = std::chrono::steady_clock::now();
    const auto timeout_time = start_time + timeout;
    const auto is_timeout_exceeded = [&]() -> bool {
        const auto now = std::chrono::steady_clock::now();
        return now > timeout_time;
    };

    std::optional<std::string> error;

    do {
        instance_.ProcessEvents();

        auto res = waitAny(timeout_time - std::chrono::steady_clock::now());

        if (!res) {
            error = res.error();
            if (stop_on_error) {
                break;
            }
        }

        bool is_done = res.value();
        if (is_done) {
            break;
        }

        std::this_thread::sleep_for(10ms);
    } while (!is_timeout_exceeded());

    if (pending_error_) {
        error = error.value_or("") + pending_error_.value();
    }

    if (is_timeout_exceeded()) {
        auto labels
            = tasks_.getWaitInfos()
              | std::views::filter([](const wgpu::FutureWaitInfo& info) {
                    return !info.completed;
                })
              | std::views::transform(
                  [](const wgpu::FutureWaitInfo& info) { return info.future; })
              | std::views::transform([this](const wgpu::Future& future) {
                    return tasks_.getFuture(future)
                        .value()
                        .future_info.error_label;
                })
              | std::views::join_with(std::string(", "))
              | std::ranges::to<std::string>();

        return "timeout exceeded for labels [" + labels + "]";
    }

    pending_error_ = std::nullopt;
    tasks_.clear();

    return error;
}

std::expected<bool, std::string> Awaiter::waitAny(
    std::chrono::nanoseconds timeout) {
    instance_.WaitAny(tasks_.size(), tasks_.getWaitInfos().data(),
                      static_cast<uint64_t>(timeout.count()));

    bool are_all_completed = true;
    for (size_t i = 0; i < tasks_.size(); i++) {
        auto task = tasks_.at(i).value();

        bool completed = task.wait_info.completed;

        are_all_completed &= completed;

#ifdef LOG_AWAITER_CALLS
        bool is_errscope_future = std::invoke([&task]() {
            return task.future_info.error_label.contains("errscope");
        });

        if (completed && !task.is_completion_logged && !is_errscope_future) {
            spdlog::debug(LOG_ID " Future completed. Label: {}",
                          task.future_info.error_label);

            task.is_completion_logged = true;
            tasks_.set(i, task);
        }
#endif
    }

    return are_all_completed;
}
