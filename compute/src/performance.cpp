#include "compute/performance.hpp"

#include <Eigen/src/Core/Matrix.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <ranges>
#include <string_view>

#include "yaml-cpp/node/emit.h"

using namespace wslam::compute;

namespace {
struct ThreadState {
    std::string id;
    // Path of the open scopes, rooted at the id; empty while none is open
    std::string scope;
};

thread_local ThreadState g_thread_state;
std::atomic<uint32_t> g_unnamed_threads{0};
}  // namespace

void PerfRecorder::setThreadId(std::string thread_id) {
    assert(thread_id.find(kDelimiter) == std::string::npos
           && "perf thread id contains the scope delimiter");
    assert(g_thread_state.scope.empty()
           && "perf thread id set while a scope is open");
    g_thread_state.id = std::move(thread_id);
}

const std::string& PerfRecorder::getThreadId() {
    if (g_thread_state.id.empty()) {
        g_thread_state.id = std::format("thread {}", g_unnamed_threads++);
    }
    return g_thread_state.id;
}

auto PerfRecorder::beginRecord(std::string_view label) -> Scope {
#ifdef NPERF
    (void)label;
    return Scope(*this, {});
#else
    std::string& scope = g_thread_state.scope;
    if (scope.empty()) {
        scope = getThreadId();
    }
    scope += kDelimiter;
    scope += label;
    return Scope(*this, Clock::now());
#endif
}

void PerfRecorder::popScope(const Scope& scope) noexcept {
#ifdef NPERF
    (void)scope;
#else
    const auto elapsed = std::chrono::duration<double, std::ratio<1, 1> >(
        Clock::now() - scope.start);

    std::string& current = g_thread_state.scope;
    {
        std::scoped_lock lock(mutex_);
        records_[current].calls.emplace_back(
            static_cast<double>(elapsed.count()));
    }

    current.resize(current.find_last_of(kDelimiter));
    // Only the thread id is left: no scope is open
    if (current.find_last_of(kDelimiter) == std::string::npos) {
        current.clear();
    }
#endif
}

void PerfRecorder::clear() {
    std::scoped_lock lock(mutex_);
    records_.clear();
}

#ifndef NPERF
namespace {
YAML::Node StatsToYaml(const PerfRecorder::Stats& stats) {
    YAML::Node node(YAML::NodeType::Map);

    if (stats.calls.empty()) {
        node["count"] = 0;
        return node;
    }

    std::vector<double> calls_sorted = stats.calls;
    std::ranges::sort(calls_sorted);

    const Eigen::Map<const Eigen::VectorXd> mapped(
        calls_sorted.data(), static_cast<Eigen::Index>(calls_sorted.size()));

    Eigen::VectorXd calls = mapped;

    const auto set = [&](std::string_view key, std::floating_point auto val) {
        node[key] = std::format("{:.3f}", val);
    };

    set("avg", calls.mean());
    set("max", calls.maxCoeff());
    set("total", calls.sum());

    std::array<std::string_view, 3> perc_labels = {
        "perc_25",
        "perc_50",
        "perc_75",
    };

    for (size_t i : std::views::iota(1UZ, 3UZ)) {
        const auto label = perc_labels.at(i);

        const auto perc = static_cast<double>(i) * 0.25;
        const auto pos = perc * static_cast<double>(calls_sorted.size());

        set(label, calls_sorted.at(static_cast<size_t>(pos)));
    }

    node["count"] = calls.size();

    return node;
}

}  // namespace
#endif

std::optional<std::string> PerfRecorder::writeFile(
    const std::filesystem::path& path) const {
#ifdef NPERF
    // Recording is compiled out: there is nothing to write
    (void)path;
    return std::nullopt;
#else
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    if (!file) {
        return std::format("opening '{}' for writing: {}", path.string(),
                           std::strerror(errno));
    }

    file << YAML::Dump(toYaml()) << '\n';

    // Flush explicitly: errors surfacing only at destruction are lost.
    if (!file.flush()) {
        return std::format("Failed to write '{}': {}", path.string(),
                           std::strerror(errno));
    }

    return std::nullopt;
#endif
}

YAML::Node PerfRecorder::toYaml() const {
    YAML::Node root(YAML::NodeType::Map);
#ifdef NPERF
    return root;
#else
    std::scoped_lock lock(mutex_);

    for (const auto& [k, v] : records_) {
        YAML::Node current = root;
        for (const auto part : std::views::split(k, kDelimiter)) {
            // reset() REBINDS the handle to the child node. Using
            // `current = current[key]` instead would call operator=,
            // which assigns the node's *contents*
            current.reset(current[std::string(std::string_view(part))]);
        }
        current["stats"] = StatsToYaml(v);
    }

    return root;
#endif
}
