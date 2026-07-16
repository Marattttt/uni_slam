#include "compute/performance.hpp"

#include <Eigen/src/Core/Matrix.h>

#include <chrono>
#include <fstream>
#include <ranges>
#include <string_view>

#include "yaml-cpp/node/emit.h"

using namespace wslam::compute;

auto PerfRecorder::beginRecord(std::string_view label) -> Scope {
    addScope(label);
    return Scope{.recorder = *this, .start = Clock::now()};
}

void PerfRecorder::addScope(std::string_view scope) {
    if (current_scope_.length() > 0) {
        current_scope_ += kDelimiter;
    }
    current_scope_ += scope;
}

void PerfRecorder::popScope(const Scope& scope) noexcept {
    const auto elapsed = std::chrono::duration<double, std::ratio<1, 1>>(
        Clock::now() - scope.start);
    records_[current_scope_].calls.emplace_back(
        static_cast<double>(elapsed.count()));

    const auto last_segment = current_scope_.find_last_of(kDelimiter);
    if (last_segment == std::string::npos) {
        current_scope_.clear();
        return;
    }
    current_scope_.resize(last_segment);
}

void PerfRecorder::clear() {
    current_scope_ = "";
    records_.clear();
}

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

std::optional<std::string> PerfRecorder::writeFile(
    const std::filesystem::path& path) const {
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
}

YAML::Node PerfRecorder::toYaml() const {
    YAML::Node root(YAML::NodeType::Map);

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
}
