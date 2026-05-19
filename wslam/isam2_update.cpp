#include "isam2_update.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
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
    if (!worker_) {
        worker_ = std::make_unique<Isam2Worker>();
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
    return s;
}

}  // namespace

std::optional<std::string> Isam2UpdatePass::drainPending() {
    if (!pending_.valid()) {
        return std::nullopt;
    }
    auto result = pending_.get();  // blocks if worker still running
    if (result.error.has_value()) {
        return result.error;
    }
    // gtsam::Values has no move-assign overload, so std::move would be a no-op.
    state_.latest_values = result.latest_values;

    assert(storage_ != nullptr);
    auto snap = Snapshot(state_);
    snap.stats.factors = result.factor_count;
    snap.stats.last_error = 0.0;  // not currently exposed by ISAM2Result

#ifndef NDEBUG
    // The optimised snapshot can lag at most one frame behind the
    // front-end's allocation counters — the in-flight submission, if any,
    // is the only outstanding gap.
    assert(snap.stats.keyframes <= state_.next_pose_id);
    assert(snap.stats.landmarks <= state_.next_landmark_id);
#endif

    storage_->set(ResourceIdentifier::MapSnapshotName, std::move(snap));
    return std::nullopt;
}

std::optional<std::string> Isam2UpdatePass::execute() {
    spdlog::info(LOG_ID " Executing");

    assert(storage_ != nullptr);
    assert(worker_);

    // Step 1: harvest the previous frame's iSAM result before submitting
    // new work. This is the synchronisation point that enforces the
    // "main thread runs at most one frame ahead of the optimiser" invariant.
    if (auto err = drainPending()) {
        return err;
    }

    if (!builder_.hasWork()) {
        spdlog::debug(LOG_ID
                      " Factor builder reported no work; nothing to submit");
        // Make sure downstream consumers always see *some* snapshot under
        // MapSnapshotName, even before iSAM has run for the first time.
        if (!storage_->has(ResourceIdentifier::MapSnapshotName)) {
            storage_->set(ResourceIdentifier::MapSnapshotName, Snapshot(state_));
        }
        return std::nullopt;
    }

    // Step 2: submit the current frame's work to the worker. We copy the
    // factor graph and values out of the builder so the builder is free to
    // clear them on the next frame while iSAM is still processing.
    pending_ = worker_->submit(Isam2Worker::Work{
        .new_factors = builder_.newFactors(),
        .new_values = builder_.newValues(),
        .extra_updates = opts_.extra_updates,
        .frame_id = frame_counter_++,
    });
    return std::nullopt;
}

std::optional<std::string> Isam2UpdatePass::flush() {
    spdlog::info(LOG_ID " flush(): draining any in-flight iSAM update");
    return drainPending();
}
