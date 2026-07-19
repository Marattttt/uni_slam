#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "compute/pass.hpp"
#include "compute/performance.hpp"

namespace wslam {
namespace compute {

constexpr std::string kStageStopExecution = "STAGE_STOP";

// A named, ordered sequence of passes. Use GpuStage for batching GPU work
class Stage {
   public:
    explicit Stage(std::string id, PerfRecorder* perf = nullptr);

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;
    Stage(Stage&&) = default;
    Stage& operator=(Stage&&) = default;
    virtual ~Stage() = default;

    [[nodiscard]] std::string getId() const;
    [[nodiscard]] virtual std::optional<std::string> initialize();
    [[nodiscard]] virtual std::optional<std::string> execute();

    void add_pass(std::unique_ptr<Pass> pass);

   protected:
    std::string id_;
    PerfRecorder* perf_;

   private:
    std::vector<std::unique_ptr<Pass>> passes_;
};
};  // namespace compute
};  // namespace wslam
