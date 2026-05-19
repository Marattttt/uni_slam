#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "common.hpp"
#include "compute.hpp"
#include "mapping_state.hpp"
#include "stage.hpp"

namespace wslam {

// Owns the long-lived MappingState (iSAM2, calibration, landmark tracks) and
// the three passes that drive the factor graph. The state is held by shared
// pointer so the passes can reference it without lifetime gymnastics.
//
// `flush_async` blocks until any in-flight iSAM2 worker job completes and
// the resulting MapSnapshot is published. Call it after the pipeline loop
// exits, before reading the final snapshot. The returned callable references
// the iSAM update pass owned by `stage`, so it must be invoked while the
// stage (and its enclosing Compute) is still alive.
struct MappingStage {
    std::shared_ptr<MappingState> state;
    compute::Stage stage;
    std::function<std::optional<std::string>()> flush_async;
};

// Factory: builds the mapping stage and returns it along with the state owner.
// Caller moves `stage` into Compute::addStage and keeps `state` alive (or
// drops it — Compute keeps the passes alive which transitively keep state
// alive via the shared_ptr inside).
[[nodiscard]] MappingStage CreateMappingStage(compute::Compute& compute,
                                              AnyBag& storage,
                                              WslamConfig config = {});

}  // namespace wslam
