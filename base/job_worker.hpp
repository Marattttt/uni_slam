#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "assert.hpp"

namespace wslam::base {

// Runs a job function on a dedicated thread, with at most one job in flight.
//
// submit() hands a unit of work over and returns at once with a future for the
// result. Backpressure is the caller's: the future of the previous submission
// must be consumed (or the job otherwise known finished) before the next
// submit(), which asserts that nothing is pending. Anything the job function
// touches belongs to the worker thread while a job runs; the caller may read
// it again once it has the job's result.
//
// An exception escaping the job function is stored in the future. The
// destructor lets a running job finish, then joins.
template <typename Work, typename Result>
class JobWorker {
   public:
    using Job = std::function<Result(Work&&)>;

    explicit JobWorker(Job job) : job_(std::move(job)) {
        thread_ = std::thread([this] { run(); });
    }

    ~JobWorker() {
        {
            std::scoped_lock lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    JobWorker(const JobWorker&) = delete;
    JobWorker& operator=(const JobWorker&) = delete;
    JobWorker(JobWorker&&) = delete;
    JobWorker& operator=(JobWorker&&) = delete;

    [[nodiscard]] std::future<Result> submit(Work work) {
        std::promise<Result> promise;
        auto future = promise.get_future();
        {
            std::scoped_lock lock(mutex_);
            WSLAM_ASSERT(!pending_.has_value() && !running_,
                         "JobWorker: previous job still in flight");
            pending_.emplace(std::move(work), std::move(promise));
        }
        condition_.notify_one();
        return future;
    }

   private:
    Job job_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<std::pair<Work, std::promise<Result>>> pending_;
    bool running_ = false;
    bool stop_ = false;

    void run() {
        while (true) {
            std::optional<std::pair<Work, std::promise<Result>>> item;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock,
                                [&] { return stop_ || pending_.has_value(); });
                if (!pending_.has_value()) {
                    return;
                }
                item = std::move(pending_);
                pending_.reset();
                running_ = true;
            }

            auto& [work, promise] = *item;
            std::optional<Result> result;
            std::exception_ptr error;
            try {
                result.emplace(job_(std::move(work)));
            } catch (...) {
                error = std::current_exception();
            }

            // Not running before the future is ready, so a caller that has
            // the result may submit again
            {
                std::scoped_lock lock(mutex_);
                running_ = false;
            }
            if (error) {
                promise.set_exception(error);
            } else {
                promise.set_value(std::move(result).value());
            }
        }
    }
};

}  // namespace wslam::base
