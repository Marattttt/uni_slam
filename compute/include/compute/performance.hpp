#pragma once

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <flat_map>
#include <string>
#include <string_view>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace wslam::compute {

// Class for recording performance of different segments of code.
// Supports nested calls: beginRecord returns a Scope, and while it is alive,
// all further calls to beginRecord will fall under that scope
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

    [[nodiscard]] Scope beginRecord(std::string_view label);
    [[nodiscard]] std::optional<std::string> writeFile(
        const std::filesystem::path& path) const;
    [[nodiscard]] YAML::Node toYaml() const;

    void clear();

   private:
    std::flat_map<std::string, Stats> records_;
    std::string current_scope_;

    void popScope(const Scope& scope) noexcept;
    void addScope(std::string_view scope);
};

}  // namespace wslam::compute
