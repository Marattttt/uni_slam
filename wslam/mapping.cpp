#include "mapping.hpp"

#include <spdlog/spdlog.h>

#include <memory>

#include "factor_builder.hpp"
#include "isam2_update.hpp"
#include "keyframe_gate.hpp"

using namespace wslam;

#define LOG_ID "[Mapping stage]"

MappingStage wslam::CreateMappingStage(compute::Compute& compute,
                                       AnyBag& storage, WslamConfig /*config*/) {
    spdlog::info(LOG_ID " Constructing mapping stage");

    const auto gpu = compute.getGPUPtr();
    auto state = std::make_shared<MappingState>();

    compute::Stage stage{"Mapping", gpu};

    auto gate = std::make_unique<KeyframeGatePass>(*state, gpu);
    gate->setStorage(storage);

    auto builder = std::make_unique<FactorBuilderPass>(*state, gpu);
    builder->setStorage(storage);

    // The iSAM2 update pass needs to read the builder's per-frame factor /
    // value containers. Hold a raw pointer (lifetimes are tied: both live in
    // the same Stage, destroyed together).
    const FactorBuilderPass* builder_ptr = builder.get();

    auto updater = std::make_unique<Isam2UpdatePass>(*state, *builder_ptr, gpu);
    updater->setStorage(storage);
    // Same lifetime argument as for builder_ptr above: the updater is added
    // to `stage` below and lives as long as that stage does.
    Isam2UpdatePass* updater_ptr = updater.get();

    stage.add_pass(std::move(gate));
    stage.add_pass(std::move(builder));
    stage.add_pass(std::move(updater));

    return MappingStage{
        .state = std::move(state),
        .stage = std::move(stage),
        .flush_async =
            [updater_ptr]() { return updater_ptr->flush(); },
    };
}
