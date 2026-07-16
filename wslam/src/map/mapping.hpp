#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "common.hpp"
#include "compute/compute.hpp"
#include "mapping_state.hpp"
#include "compute/stage.hpp"

namespace wslam {

// Holds the three passes that drive the factor graph. The long-lived
// MappingState (iSAM2, calibration, landmark tracks) is owned jointly by the
// passes themselves via a shared_ptr, so the stage keeps it alive for as long
// as Compute keeps the passes — no separate owner handle is needed here.
//
// `flush_async` blocks until any in-flight iSAM2 worker job completes and
// the resulting MapSnapshot is published. Call it after the pipeline loop
// exits, before reading the final snapshot. The returned callable references
// the iSAM update pass owned by `stage`, so it must be invoked while the
// stage (and its enclosing Compute) is still alive.
struct MappingStage {
    compute::Stage stage;
    std::function<std::optional<std::string>()> flush_async;
};

// Factory: builds the mapping stage. Caller moves `stage` into
// Compute::addStage; the MappingState lives as long as the passes do (shared
// ownership inside each pass).
[[nodiscard]] MappingStage CreateMappingStage(compute::Compute& compute,
                                              AnyBag& storage,
                                              const WslamConfig& config = {});

}  // namespace wslam
