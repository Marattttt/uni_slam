#pragma once

#include <memory>

#include "common.hpp"
#include "compute.hpp"
#include "mapping_state.hpp"
#include "stage.hpp"

namespace wslam {

// Owns the long-lived MappingState (iSAM2, calibration, landmark tracks) and
// the three passes that drive the factor graph. The state is held by shared
// pointer so the passes can reference it without lifetime gymnastics.
struct MappingStage {
    std::shared_ptr<MappingState> state;
    compute::Stage stage;
};

// Factory: builds the mapping stage and returns it along with the state owner.
// Caller moves `stage` into Compute::addStage and keeps `state` alive (or
// drops it — Compute keeps the passes alive which transitively keep state
// alive via the shared_ptr inside).
[[nodiscard]] MappingStage CreateMappingStage(compute::Compute& compute,
                                              AnyBag& storage,
                                              WslamConfig config = {});

}  // namespace wslam
