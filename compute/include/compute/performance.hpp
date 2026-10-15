#pragma once

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <flat_map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace wslam::compute {

// Class for recording performance of different segments of code.
// Supports nested calls: beginRecord returns a Scope, and while it is alive,
// all further calls to beginRecord on the same thread will fall under that
// scope
//
// Safe to record into from several threads. Every thread's records are rooted
// at its thread id (setThreadId, or "thread <n>" for a thread that never set
// one), so each thread gets a tree of its own. The open scopes are per thread
// and shared by every recorder on it
//
// Defining NPERF makes calls of this class essentially no-op
class PerfRecorder {
    static constexpr std::string_view kDelimiter = "$";

   public:
    using Clock = std::chrono::steady_clock;
    using Duration = Clock::duration;

    struct Scope {
        PerfRecorder& recorder;
        Clock::time_point start;

        Scope(PerfRecorder& recorder, Clock::time_point start)
            : recorder(recorder), start(start) {}
        // A moved-from scope must not pop: wrapping a scope in an optional
        // otherwise pops it twice, once right away and once at the end
        Scope(Scope&& other) noexcept
            : recorder(other.recorder), start(std::exchange(other.start, {})) {}
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope& operator=(Scope&&) = delete;

        ~Scope() {
            // Ensure that a non-empty scope is the only one calling popScope
            if (start.time_since_epoch().count() > 0) {
                recorder.popScope(*this);
            }
        };
    };

    struct Stats {
        std::vector<double> calls;
    };

    // Names the calling thread's root in every recorder. Call it before the
    // thread records anything; the id must not contain the delimiter '$'
    static void setThreadId(std::string thread_id);
    [[nodiscard]] static const std::string& getThreadId();

    [[nodiscard]] Scope beginRecord(std::string_view label);
    [[nodiscard]] std::optional<std::string> writeFile(
        const std::filesystem::path& path) const;
    [[nodiscard]] YAML::Node toYaml() const;

    // Drops the records of every thread. Scopes still open record when they
    // close
    void clear();

   private:
    mutable std::mutex mutex_;
    std::flat_map<std::string, Stats> records_;

    void popScope(const Scope& scope) noexcept;
};

}  // namespace wslam::compute
