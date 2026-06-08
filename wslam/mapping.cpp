#include "mapping.hpp"

#include <spdlog/spdlog.h>

#include <memory>

#include "factor_builder.hpp"
#include "isam2_update.hpp"
#include "keyframe_gate.hpp"

using namespace wslam;

#define LOG_ID "[Mapping stage]"

MappingStage wslam::CreateMappingStage(compute::Compute& compute,
                                       AnyBag& storage,
                                       const WslamConfig& config) {
    spdlog::info(LOG_ID " Constructing mapping stage");

    auto state = std::make_shared<MappingState>();

    compute::Stage stage{"Mapping", compute};

    auto gate = std::make_unique<KeyframeGatePass>(state);
    gate->setStorage(storage);

    auto builder = std::make_unique<FactorBuilderPass>(state);
    builder->setStorage(storage);

    // The iSAM2 update pass needs to read the builder's per-frame factor /
    // value containers. Hold a raw pointer (lifetimes are tied: both live in
    // the same Stage, destroyed together).
    const FactorBuilderPass* builder_ptr = builder.get();

    // Headless runs have no consumer of intermediate snapshots — the map
    // export only reads the post-flush one — so rebuild the (O(map size))
    // snapshot rarely. The GUI reads it every frame.
    Isam2UpdatePass::Opts updater_opts{};
    updater_opts.snapshot_every_n_drains = config.enable_gui ? 1 : 25;

    auto updater = std::make_unique<Isam2UpdatePass>(std::move(state),
                                                     *builder_ptr, updater_opts);
    updater->setStorage(storage);
    // Same lifetime argument as for builder_ptr above: the updater is added
    // to `stage` below and lives as long as that stage does.
    Isam2UpdatePass* updater_ptr = updater.get();

    // Drain pass goes FIRST so the keyframe gate and factor builder see
    // up-to-date pose / smart-factor estimates from the previous iSAM
    // round. Without this, the factor builder would read stale
    // smart_factor_indices and schedule removal of factor indices iSAM
    // had already replaced — yielding VariableIndex::remove failures.
    auto drainer = std::make_unique<Isam2DrainPass>(*updater_ptr);

    stage.add_pass(std::move(drainer));
    stage.add_pass(std::move(gate));
    stage.add_pass(std::move(builder));
    stage.add_pass(std::move(updater));

    return MappingStage{
        .stage = std::move(stage),
        .flush_async =
            [updater_ptr]() { return updater_ptr->flush(); },
    };
}
