#include "isam2_update.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <spdlog/spdlog.h>

#include <utility>

#include "common.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[ISAM2 Update pass]"

Isam2UpdatePass::Isam2UpdatePass(MappingState& state,
                                 const FactorBuilderPass& builder,
                                 std::shared_ptr<compute::GPU> gpu, Opts opts)
    : compute::Pass(std::move(gpu)),
      state_(state),
      builder_(builder),
      opts_(opts) {}

Isam2UpdatePass::Isam2UpdatePass(MappingState& state,
                                 const FactorBuilderPass& builder,
                                 std::shared_ptr<compute::GPU> gpu)
    : Isam2UpdatePass(state, builder, std::move(gpu), Opts{}) {}

std::string Isam2UpdatePass::getId() const { return LOG_ID; }

std::optional<std::string> Isam2UpdatePass::initialize() {
    spdlog::info(LOG_ID " Initializing (extra_updates={})", opts_.extra_updates);
    if (storage_ == nullptr) {
        return "isam2 update: storage not set before initialize()";
    }
    if (!state_.isam) {
        gtsam::ISAM2Params params;
        // Reasonable defaults for incremental visual SLAM: gauss-newton with
        // a small relinearization threshold so newly added landmarks settle
        // quickly without the cost of full QR re-elimination every frame.
        params.optimizationParams = gtsam::ISAM2GaussNewtonParams();
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip = 1;
        state_.isam = std::make_unique<gtsam::ISAM2>(params);
    }
    return std::nullopt;
}

namespace {

// Collects the current estimate into a portable snapshot. Keeps the GTSAM
// dependency contained — downstream consumers (e.g. the GUI pass) deal in
// Eigen + IDs only.
MapSnapshot Snapshot(const MappingState& state) {
    MapSnapshot s;
    s.keyframes.reserve(state.latest_values.size());
    s.landmarks.reserve(state.latest_values.size());

    for (const auto& kv : state.latest_values) {
        const auto key = kv.key;
        const gtsam::Symbol sym(key);
        if (sym.chr() == MappingState::kPoseChar) {
            const auto& pose = state.latest_values.at<gtsam::Pose3>(key);
            s.keyframes.push_back(KeyframePose{
                .id = PoseId{sym.index()},
                .R_world_cam = pose.rotation().matrix(),
                .t_world_cam = pose.translation(),
            });
        } else if (sym.chr() == MappingState::kLandmarkChar) {
            const auto& pt = state.latest_values.at<gtsam::Point3>(key);
            s.landmarks.push_back(LandmarkEstimate{
                .id = LandmarkId{sym.index()},
                .position_world = pt,
            });
        }
    }

    s.stats.keyframes = s.keyframes.size();
    s.stats.landmarks = s.landmarks.size();
    if (state.isam) {
        s.stats.factors = state.isam->getFactorsUnsafe().size();
    }
    return s;
}

}  // namespace

std::optional<std::string> Isam2UpdatePass::execute() {
    spdlog::info(LOG_ID " Executing");

    assert(storage_ != nullptr);

    if (!builder_.hasWork()) {
        spdlog::debug(LOG_ID
                      " Factor builder reported no work; publishing previous "
                      "snapshot");
        storage_->set(ResourceIdentifier::MapSnapshotName, Snapshot(state_));
        return std::nullopt;
    }

    assert(state_.isam);

    try {
        spdlog::debug(LOG_ID
                      " submitting to iSAM2: new_factors={}, new_values={}",
                      builder_.newFactors().size(), builder_.newValues().size());
        const auto result = state_.isam->update(builder_.newFactors(),
                                                builder_.newValues());
        for (uint32_t i = 0; i < opts_.extra_updates; ++i) {
            state_.isam->update();
        }
        state_.latest_values = state_.isam->calculateEstimate();

        spdlog::info(
            LOG_ID
            " iSAM2 update OK: variables_relinearized={}, factors_recalc={}, "
            "total_factors={}, total_values={}",
            result.variablesRelinearized, result.factorsRecalculated,
            state_.isam->getFactorsUnsafe().size(), state_.latest_values.size());
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
        return std::format(
            "iSAM2 update: indeterminate linear system at key '{}' (chr "
            "'{}' index {}). Likely cause: a variable has no constraints "
            "yet. what(): {}",
            gtsam::DefaultKeyFormatter(e.nearbyVariable()),
            static_cast<char>(gtsam::Symbol(e.nearbyVariable()).chr()),
            gtsam::Symbol(e.nearbyVariable()).index(), e.what());
    } catch (const std::exception& e) {
        return std::format("iSAM2 update threw [{}]: {}", typeid(e).name(),
                           e.what());
    }

    auto snap = Snapshot(state_);
    snap.stats.last_error = 0.0;  // not currently exposed by ISAM2Result

#ifndef NDEBUG
    // The snapshot's keyframe count must match the number of accepted
    // keyframes we've recorded on the mapping state.
    assert(snap.stats.keyframes == state_.next_pose_id);
    assert(snap.stats.landmarks == state_.next_landmark_id);
#endif

    storage_->set(ResourceIdentifier::MapSnapshotName, std::move(snap));
    return std::nullopt;
}
