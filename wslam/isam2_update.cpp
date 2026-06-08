#include "isam2_update.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <spdlog/spdlog.h>

#include <utility>

#include "common.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[ISAM2 Update pass]"

Isam2UpdatePass::Isam2UpdatePass(std::shared_ptr<MappingState> state,
                                 const FactorBuilderPass& builder, Opts opts)
    : state_(std::move(state)), builder_(builder), opts_(opts) {}

Isam2UpdatePass::Isam2UpdatePass(std::shared_ptr<MappingState> state,
                                 const FactorBuilderPass& builder)
    : Isam2UpdatePass(std::move(state), builder, Opts{}) {}

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
//
// Keyframes come straight from `latest_values` (pose variables are still
// in the graph). Landmarks come from `state.smart_factors`: each smart
// factor's `point()` accessor returns the triangulated landmark at the
// current poses' operating point. Only VALID triangulations are kept —
// DEGENERATE / OUTLIER / BEHIND_CAMERA / FAR_POINT entries are dropped.
MapSnapshot Snapshot(const MappingState& state) {
    MapSnapshot s;
    s.keyframes.reserve(state.latest_values.size());
    s.landmarks.reserve(state.smart_factors.size());

    for (const auto& kv : state.latest_values) {
        const gtsam::Symbol sym(kv.key);
        if (sym.chr() != MappingState::kPoseChar) {
            continue;
        }
        const auto& pose = state.latest_values.at<gtsam::Pose3>(kv.key);
        const PoseId pose_id{sym.index()};
        // Every accepted pose was registered by the keyframe gate before
        // it ever reached iSAM, so a miss here is a state inconsistency.
        // Fall back to 0 (rather than throwing) so the snapshot still
        // publishes; downstream consumers can detect the missing timestamp.
        const auto timestamp_it = state.keyframe_timestamps_ns.find(pose_id);
        const uint64_t timestamp_ns
            = timestamp_it == state.keyframe_timestamps_ns.end()
                  ? 0
                  : timestamp_it->second;
        s.keyframes.push_back(KeyframePose{
            .id = pose_id,
            .timestamp_ns = timestamp_ns,
            .R_world_cam = pose.rotation().matrix(),
            .t_world_cam = pose.translation(),
        });
    }

    for (const auto& [id, factor] : state.smart_factors) {
        // Skip factors whose observation poses haven't all been
        // optimised yet. This happens at the start of a frame: the
        // factor builder has just appended observations at x_curr to
        // smart factors for re-observed landmarks, but latest_values
        // still reflects the previous drain (which only knows about
        // poses up to x_prev). Calling point() on such a factor would
        // throw ValuesKeyDoesNotExist when it dereferences x_curr.
        bool all_keys_present = true;
        for (const auto& k : factor->keys()) {
            if (!state.latest_values.exists(k)) {
                all_keys_present = false;
                break;
            }
        }
        if (!all_keys_present) {
            continue;
        }
        const auto tri = factor->point(state.latest_values);
        if (tri.valid()) {
            s.landmarks.push_back(LandmarkEstimate{
                .id = id,
                .position_world = *tri,
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
    state_->latest_values = result.latest_values;

    // Mirror optimised pose estimates back into predicted_values so the
    // keyframe-gate's pose chain uses bundle-adjusted estimates. With
    // smart factors, landmarks are no longer graph variables, so only
    // pose keys are mirrored. V/B keys are handled below via the
    // dedicated last_velocity / last_bias fields.
    for (const auto& kv : state_->latest_values) {
        if (gtsam::Symbol(kv.key).chr() != MappingState::kPoseChar) {
            continue;
        }
        assert(state_->predicted_values.exists(kv.key));
        state_->predicted_values.update(kv.key, kv.value);
    }

    // Back-fill MappingState::smart_factor_indices using the FactorIndex
    // values iSAM2 just assigned. pending_smart_factor_positions_ pairs
    // each landmark id with its position in the previous frame's
    // new_factors graph; result.new_factor_indices is parallel to that
    // graph and gives the assigned FactorIndex per position.
    for (const auto& [pos, lm_id] : pending_smart_factor_positions_) {
        assert(pos < result.new_factor_indices.size());
        state_->smart_factor_indices.insert_or_assign(
            lm_id, result.new_factor_indices[pos]);
    }
    pending_smart_factor_positions_.clear();

    // Propagate the latest velocity and bias estimates so the factor
    // builder can seed the next keyframe's V/B with the freshest
    // values. We pick the entries belonging to the highest pose id
    // present — that's the one the next CombinedImuFactor will chain off.
    if (state_->next_pose_id > 0) {
        const auto last_id = PoseId{state_->next_pose_id - 1};
        const auto vk = MappingState::velocityKey(last_id);
        const auto bk = MappingState::biasKey(last_id);
        if (state_->latest_values.exists(vk)) {
            state_->last_velocity = state_->latest_values.at<gtsam::Vector3>(vk);
        }
        if (state_->latest_values.exists(bk)) {
            state_->last_bias
                = state_->latest_values.at<gtsam::imuBias::ConstantBias>(bk);
        }
    }

    assert(storage_ != nullptr);
    last_factor_count_ = result.factor_count;

    // Building the snapshot triangulates every smart factor ever created
    // — O(map size) on the main thread — so headless configurations only
    // do it every Nth drain. flush() publishes unconditionally, so the
    // exported map always reflects the final optimisation.
    ++drains_since_snapshot_;
    if (drains_since_snapshot_ >= opts_.snapshot_every_n_drains) {
        publishSnapshot();
    }
    return std::nullopt;
}

void Isam2UpdatePass::publishSnapshot() {
    assert(storage_ != nullptr);
    drains_since_snapshot_ = 0;

    auto snap = Snapshot(*state_);
    snap.stats.factors = last_factor_count_;
    snap.stats.last_error = 0.0;  // not currently exposed by ISAM2Result

#ifndef NDEBUG
    // The optimised snapshot can lag at most one frame behind the
    // front-end's allocation counters — the in-flight submission, if any,
    // is the only outstanding gap. Landmarks may be < next_landmark_id
    // because degenerate smart-factor triangulations are filtered out.
    assert(snap.stats.keyframes <= state_->next_pose_id);
    assert(snap.stats.landmarks <= state_->smart_factors.size());
#endif

    storage_->set(ResourceIdentifier::MapSnapshotName, std::move(snap));
}

std::optional<std::string> Isam2UpdatePass::execute() {
    spdlog::info(LOG_ID " Executing");

    assert(storage_ != nullptr);
    assert(worker_);

    // The companion Isam2DrainPass at the start of the mapping stage
    // already harvested the previous frame's result, so we just submit
    // this frame's work below.

    if (!builder_.hasWork()) {
        spdlog::debug(LOG_ID
                      " Factor builder reported no work; nothing to submit");
        // Make sure downstream consumers always see *some* snapshot under
        // MapSnapshotName, even before iSAM has run for the first time.
        if (!storage_->has(ResourceIdentifier::MapSnapshotName)) {
            storage_->set(ResourceIdentifier::MapSnapshotName,
                          Snapshot(*state_));
        }
        return std::nullopt;
    }

    // Step 2: submit the current frame's work to the worker. We copy the
    // factor graph and values out of the builder so the builder is free to
    // clear them on the next frame while iSAM is still processing. Also
    // snapshot the smart-factor positions so drainPending can back-fill
    // smart_factor_indices when iSAM returns the new FactorIndices.
    pending_smart_factor_positions_ = builder_.smartFactorPositions();
    pending_ = worker_->submit(Isam2Worker::Work{
        .new_factors = builder_.newFactors(),
        .new_values = builder_.newValues(),
        .remove_factor_indices = builder_.removeIndices(),
        .extra_updates = opts_.extra_updates,
        .frame_id = frame_counter_++,
    });
    return std::nullopt;
}

std::optional<std::string> Isam2UpdatePass::flush() {
    spdlog::info(LOG_ID " flush(): draining any in-flight iSAM update");
    if (auto err = drainPending()) {
        return err;
    }
    // Consumers (e.g. ExportMap) read the snapshot right after flush;
    // bypass the drain throttle so it reflects every applied update.
    publishSnapshot();
    return std::nullopt;
}
