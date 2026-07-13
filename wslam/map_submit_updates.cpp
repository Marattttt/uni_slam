#include "map_submit_updates.hpp"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "assert.hpp"
#include "common.hpp"
#include "isam2_worker.hpp"
#include "map_common.hpp"
#include "models.hpp"

using namespace wslam;
using namespace wslam::map;

#define LOG_ID "[Submit Updates pass]"

std::string SubmitUpdatesPass::getId() const { return LOG_ID; }

std::optional<std::string> SubmitUpdatesPass::initialize() {
    spdlog::info(LOG_ID
                 " Initializing (relin_thresh={}, factorization={}, "
                 "snapshot_every_n={})",
                 opts_.relinearize_threshold,
                 opts_.use_cholesky ? "CHOLESKY" : "QR",
                 opts_.snapshot_every_n_keyframes);

    worker_ = std::make_unique<Isam2Worker>(Isam2Worker::Params{
        .relinearize_threshold = opts_.relinearize_threshold,
        .use_cholesky = opts_.use_cholesky,
    });

    publishSnapshot();
    return {};
}

namespace {

/// Collect the current estimate into a portable, GTSAM-free MapSnapshot.
///
/// Keyframe poses are read straight from the optimised `latest` Values.
/// Landmark positions come from each smart factor's internal triangulation
/// evaluated at those poses — smart factors marginalise the 3D point out of the
/// graph, so this is the only place the point becomes explicit again. Only
/// well-conditioned triangulations are kept (degenerate / behind-camera /
/// too-far points report `!valid()` and are dropped).
///
/// @param shared        persistent mapping state (smart factors, timestamps).
/// @param latest        optimised estimate for every pose / velocity / bias.
/// @param factor_count  total iSAM2 factors, recorded into snapshot stats.
MapSnapshot BuildSnapshot(const MappingSharedBindings& shared,
                          const gtsam::Values& latest,
                          std::size_t factor_count) {
    MapSnapshot snapshot;
    snapshot.keyframes.reserve(shared.pose_timestamps.size());
    snapshot.landmarks.reserve(shared.smart_factors.size());

    // --- Keyframe poses ---
    for (const auto& key_value : latest) {
        const gtsam::Symbol symbol(key_value.key);
        if (symbol.chr() != GtsamIdentifiers::kPoseCh) {
            continue;  // skip velocity / bias variables
        }
        const auto& pose = latest.at<gtsam::Pose3>(key_value.key);
        const std::size_t index = symbol.index();

        // pose_timestamps is keyed by the map-local PoseId, whereas the
        // snapshot's KeyframePose carries the models.hpp PoseId. Both wrap the
        // same integer index; a miss means the gate never recorded a timestamp
        // (a state inconsistency), so fall back to 0 rather than throwing so
        // the snapshot still publishes.
        const auto timestamp_it
            = shared.pose_timestamps.find(map::PoseId{index});
        const uint64_t timestamp_ns
            = timestamp_it == shared.pose_timestamps.end()
                  ? 0
                  : timestamp_it->second;

        snapshot.keyframes.push_back(KeyframePose{
            .id = wslam::PoseId{index},
            .timestamp_ns = timestamp_ns,
            .R_world_cam = pose.rotation().matrix(),
            .t_world_cam = pose.translation(),
        });
    }

    // --- Landmark positions ---
    for (const auto& [landmark_id, factor] : shared.smart_factors) {
        // A smart factor triangulates from the poses that observed it. Skip it
        // if any of those poses is missing from `latest`: calling point() on
        // such a factor dereferences an unknown key and throws. In the
        // synchronous design every submitted pose is optimised before we reach
        // here, so this is defensive rather than load-bearing.
        const bool all_keys_present = std::ranges::all_of(
            factor->keys(),
            [&latest](const gtsam::Key key) { return latest.exists(key); });
        if (!all_keys_present) {
            continue;
        }

        // TriangulationResult derives from std::optional<Point3> and populates
        // it *only* for a VALID result — DEGENERATE / OUTLIER / BEHIND_CAMERA /
        // FAR_POINT leave the optional empty — so an engaged optional is
        // exactly a valid triangulation. Guarding on has_value() (rather than
        // the equivalent valid()) keeps the value() access below provably
        // checked.
        const gtsam::TriangulationResult triangulation = factor->point(latest);
        if (triangulation.has_value()) {
            snapshot.landmarks.push_back(LandmarkEstimate{
                .id = wslam::LandmarkId{landmark_id.v},
                .position_world = triangulation.value(),
            });
        }
    }

    snapshot.stats.keyframes = snapshot.keyframes.size();
    snapshot.stats.landmarks = snapshot.landmarks.size();
    snapshot.stats.factors = factor_count;
    return snapshot;
}

}  // namespace

void SubmitUpdatesPass::publishSnapshot() {
    keyframes_since_snapshot_ = 0;

    auto snapshot = BuildSnapshot(shared_, latest_values_, last_factor_count_);

    WSLAM_ASSERT(snapshot.stats.keyframes <= shared_.keyframes_processed,
                 "snapshot has {} keyframes but only {} were accepted",
                 snapshot.stats.keyframes, shared_.keyframes_processed);

    storage_.set(ResourceIdentifier::MapSnapshotName, std::move(snapshot));
}

std::optional<std::string> SubmitUpdatesPass::execute() {
    spdlog::info(LOG_ID " Executing");
    WSLAM_ASSERT(worker_ != nullptr, "worker must be created in initialize()");
    WSLAM_ASSERT(!pending_.valid(),
                 "iSAM2 worker thread must finish before this pass");

    auto bundle_opt
        = storage_.take<FactorBundle>(ResourceIdentifier::FactorBundleName);
    if (!bundle_opt) {
        return "no FactorBundle in storage";
    }
    FactorBundle bundle = std::move(bundle_opt).value();

    pending_pose_id_ = shared_.last_pose;
    pending_smart_factor_positions_ = std::move(bundle.smart_factor_positions);

    spdlog::debug(LOG_ID " submitting x{}: {} factors, {} values, {} removals",
                  pending_pose_id_.v, bundle.new_factors.size(),
                  bundle.new_values.size(), bundle.remove_indices.size());

    // Hand off the work and return WITHOUT blocking. drainPending() awaits the
    // future at the top of the next accepted keyframe's mapping stage; the
    // front-end-only frames until then overlap the solve.
    pending_ = worker_->submit(Isam2Worker::Work{
        .new_factors = std::move(bundle.new_factors),
        .new_values = std::move(bundle.new_values),
        .remove_factor_indices = std::move(bundle.remove_indices),
        .extra_updates = opts_.extra_updates,
        .frame_id = pending_pose_id_.v,
    });

    return {};
}

std::optional<std::string> SubmitUpdatesPass::drainPending() {
    // Nothing submitted or already drained
    if (!pending_.valid()) {
        return {};
    }

    Isam2Worker::Result result = pending_.get();
    if (result.error.has_value()) {
        return "isam2 update: " + std::move(result.error).value();
    }

    for (const auto& [key, value] : latest_values_) {
        if (gtsam::Symbol(key).chr() != GtsamIdentifiers::kPoseCh) {
            continue;
        }
        WSLAM_ASSERT(shared_.predicted_values.exists(key),
                     "every optimised pose was inserted into predicted_values "
                     "by FilterKeyframePass at acceptance");
        shared_.predicted_values.update(key, value);
    }

    for (const auto& [position, landmark_id] :
         pending_smart_factor_positions_) {
        WSLAM_ASSERT(position < result.new_factor_indices.size(),
                     "smart-factor position {} out of range of {} returned "
                     "indices",
                     position, result.new_factor_indices.size());
        shared_.smart_factor_indices.insert_or_assign(
            landmark_id, result.new_factor_indices[position]);
    }

    latest_values_ = result.latest_values;
    const PoseId drained_id = pending_pose_id_;

    const gtsam::Key velocity_key = GtsamIdentifiers::Velocity(drained_id);
    const gtsam::Key bias_key = GtsamIdentifiers::Bias(drained_id);

    WSLAM_ASSERT(
        latest_values_.exists(velocity_key) && latest_values_.exists(bias_key),
        "the drained keyframe's velocity and bias were submitted and "
        "must survive the update");

    shared_.last_velocity = latest_values_.at<gtsam::Vector3>(velocity_key);
    shared_.last_bias
        = latest_values_.at<gtsam::imuBias::ConstantBias>(bias_key);

    last_factor_count_ = result.factor_count;

    spdlog::debug(LOG_ID " drained x{}: {} factors total", drained_id.v,
                  result.factor_count);

    if (opts_.snapshot_every_n_keyframes > 0
        && ++keyframes_since_snapshot_ >= opts_.snapshot_every_n_keyframes) {
        publishSnapshot();
    }

    return {};
}

std::optional<std::string> SubmitUpdatesPass::flush() {
    // DrainUpdatesPass never runs for the final keyframe (no next keyframe
    // triggers it), so absorb that last in-flight optimisation here before
    // publishing. A no-op if nothing is pending.
    if (auto err = drainPending()) {
        return err;
    }

    // Guarantee the exported snapshot reflects the final keyframe even when the
    // in-loop cadence (snapshot_every_n_keyframes) skipped it.
    //
    // NOTE: a final global batch re-optimisation is deliberately NOT run. On
    // monocular-inertial sequences the scale-free smart-projection factors
    // vastly outnumber the scale-bearing IMU factors, so a free global solve
    // collapses metric scale (measured Umeyama scale 1.13 -> 0.001 on V101, see
    // benchmarks/ACCURACY_ANALYSIS.md). The sequential incremental estimate
    // resists this because each keyframe is anchored step-by-step.
    spdlog::info(LOG_ID " flush(): publishing final snapshot");
    publishSnapshot();
    return {};
}

std::string DrainUpdatesPass::getId() const { return "[Drain Updates pass]"; }

std::optional<std::string> DrainUpdatesPass::initialize() { return {}; }

std::optional<std::string> DrainUpdatesPass::execute() {
    spdlog::info("[Drain Updates pass] Executing");
    return target_.drainPending();
}
