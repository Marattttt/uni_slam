#include "submit_updates.hpp"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "models.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;

#define LOG_ID "[Submit Smoother pass]"

namespace {
[[nodiscard]] bool IsPoseKey(gtsam::Key key) {
    return gtsam::Symbol(key).chr() == wslam::map::GtsamIdentifiers::kPoseCh;
}
}  // namespace

std::string SubmitSmootherPass::getId() const { return LOG_ID; }

std::optional<std::string> SubmitSmootherPass::initialize() {
    spdlog::info(LOG_ID " Initializing (snapshot_every_n={})",
                 opts_.snapshot_every_n_keyframes);

    worker_ = std::make_unique<SmootherWorker>(opts_.smoother);

    publishSnapshot();
    return std::nullopt;
}

std::optional<std::string> SubmitSmootherPass::execute() {
    spdlog::info(LOG_ID " Executing");
    WSLAM_ASSERT(worker_ != nullptr, "worker is created in initialize()");
    WSLAM_ASSERT(!pending_.valid(),
                 "the previous update must be drained before submitting");

    auto bundle
        = storage_.take<FactorBundle>(ResourceIdentifier::FactorBundleName);
    if (!bundle) {
        return "no FactorBundle in storage";
    }

    pending_pose_id_ = shared_.last_pose;
    pending_ = worker_->submit(SmootherWorker::Work{
        .bundle = std::move(bundle).value(),
        .keyframe_id = pending_pose_id_.v,
    });

    return std::nullopt;
}

std::optional<std::string> SubmitSmootherPass::drainPending() {
    if (!pending_.valid()) {
        return std::nullopt;
    }

    SmootherWorker::Result result = pending_.get();
    if (result.error) {
        return "smoother update of x" + std::to_string(pending_pose_id_.v)
               + ": " + std::move(result.error).value();
    }

    archiveMarginalisedPoses(result.latest_values);
    reconcileSmartFactors(result);

    latest_values_ = std::move(result.latest_values);
    last_factor_count_ = result.factor_count;

    for (const auto& [key, value] : latest_values_) {
        if (!IsPoseKey(key)) {
            continue;
        }
        WSLAM_ASSERT(shared_.predicted_values.exists(key),
                     "every optimised pose was predicted at acceptance");
        shared_.predicted_values.update(key, value);
    }

    const gtsam::Key velocity_key = Velocity(pending_pose_id_);
    const gtsam::Key bias_key = Bias(pending_pose_id_);
    WSLAM_ASSERT(
        latest_values_.exists(velocity_key) && latest_values_.exists(bias_key),
        "the newest keyframe's velocity and bias are inside the window");
    shared_.last_drained_pose = pending_pose_id_;
    shared_.last_velocity = latest_values_.at<gtsam::Vector3>(velocity_key);
    trackVelocityError();
    shared_.last_bias
        = latest_values_.at<gtsam::imuBias::ConstantBias>(bias_key);

    spdlog::debug(LOG_ID " drained x{}: {} factors, {} values",
                  pending_pose_id_.v, last_factor_count_,
                  latest_values_.size());

    if (opts_.snapshot_every_n_keyframes > 0
        && ++keyframes_since_snapshot_ >= opts_.snapshot_every_n_keyframes) {
        publishSnapshot();
    }

    return std::nullopt;
}

void SubmitSmootherPass::trackVelocityError() {
    // Weight of the newest error in the running variance
    constexpr double kSmoothing = 0.2;

    const auto predicted = shared_.predicted_velocities.find(pending_pose_id_);
    if (predicted == shared_.predicted_velocities.end()) {
        // The first keyframe has no IMU-predicted velocity
        return;
    }
    const Eigen::Vector3d error = shared_.last_velocity - predicted->second;
    shared_.predicted_velocities.erase(shared_.predicted_velocities.begin(),
                                       std::next(predicted));

    // Per-axis variance of this drain's error
    const double variance = error.squaredNorm() / 3.0;
    shared_.velocity_error_variance
        = shared_.velocity_error_samples == 0
              ? variance
              : (1.0 - kSmoothing) * shared_.velocity_error_variance
                    + kSmoothing * variance;
    ++shared_.velocity_error_samples;

    spdlog::debug(LOG_ID
                  " x{}: velocity prediction off by {:.3f} m/s; "
                  "running sigma {:.3f} m/s",
                  pending_pose_id_.v, error.norm(),
                  std::sqrt(shared_.velocity_error_variance));
}

void SubmitSmootherPass::archiveMarginalisedPoses(
    const gtsam::Values& updated) {
    size_t archived = 0;
    for (const auto& [key, value] : latest_values_) {
        if (!IsPoseKey(key) || updated.exists(key)) {
            continue;
        }

        const PoseId id{gtsam::Symbol(key).index()};
        shared_.archived_poses.insert_or_assign(id, value.cast<gtsam::Pose3>());
        shared_.poses_archived_since_feed.push_back(id);
        if (shared_.predicted_values.exists(key)) {
            shared_.predicted_values.erase(key);
        }
        ++archived;
    }

    if (archived > 0) {
        spdlog::debug(LOG_ID " archived {} keyframes ({} total)", archived,
                      shared_.archived_poses.size());
    }
}

void SubmitSmootherPass::reconcileSmartFactors(
    const SmootherWorker::Result& result) {
    std::vector<LandmarkId> marginalised;

    for (const auto& [id, factor] : shared_.smart_factors) {
        const auto live = result.live_slots.find(factor.get());
        if (live != result.live_slots.end()) {
            shared_.smart_factor_slots.insert_or_assign(id, live->second);
        } else {
            marginalised.push_back(id);
        }
    }

    size_t archived = 0;
    for (const LandmarkId& id : marginalised) {
        const SmartFactor::shared_ptr factor = shared_.smart_factors.at(id);

        // The factor's poses are still in the window or were archived by
        // this drain. Out of the smoother, the factor is the main thread's
        gtsam::Values poses;
        for (const gtsam::Key key : factor->keys()) {
            if (result.latest_values.exists(key)) {
                poses.insert(key, result.latest_values.at(key));
            } else if (const auto archived_pose = shared_.archived_poses.find(
                           PoseId{gtsam::Symbol(key).index()});
                       archived_pose != shared_.archived_poses.end()) {
                poses.insert(key, archived_pose->second);
            }
        }

        if (poses.size() == factor->keys().size()) {
            const gtsam::TriangulationResult triangulation
                = factor->point(poses);
            if (triangulation.has_value()) {
                shared_.archived_landmarks.insert_or_assign(
                    id, triangulation.value());
                shared_.landmarks_archived_since_feed.push_back(id);
                ++archived;
            }
        }

        shared_.smart_factors.erase(id);
        shared_.smart_factor_slots.erase(id);
    }

    if (!marginalised.empty()) {
        spdlog::debug(LOG_ID
                      " {} landmarks left the window, {} archived with a "
                      "valid triangulation ({} in the archive)",
                      marginalised.size(), archived,
                      shared_.archived_landmarks.size());
    }
}

void SubmitSmootherPass::publishSnapshot() {
    keyframes_since_snapshot_ = 0;

    MapSnapshot snapshot;

    const auto timestamp_of = [&](const PoseId& id) -> uint64_t {
        const auto it = shared_.pose_timestamps.find(id);
        return it == shared_.pose_timestamps.end() ? 0 : it->second;
    };
    const auto add_keyframe = [&](const PoseId& id, const gtsam::Pose3& pose) {
        snapshot.keyframes.push_back(KeyframePose{
            .id = wslam::PoseId{id.v},
            .timestamp_ns = timestamp_of(id),
            .R_world_cam = pose.rotation().matrix(),
            .t_world_cam = pose.translation(),
        });
    };

    for (const auto& [id, pose] : shared_.archived_poses) {
        add_keyframe(id, pose);
    }
    for (const auto& [key, value] : latest_values_) {
        if (IsPoseKey(key)) {
            add_keyframe(PoseId{gtsam::Symbol(key).index()},
                         value.cast<gtsam::Pose3>());
        }
    }
    std::ranges::sort(snapshot.keyframes, {},
                      [](const KeyframePose& pose) { return pose.id.v; });

    for (const auto& [id, position] : shared_.archived_landmarks) {
        snapshot.landmarks.push_back(LandmarkEstimate{
            .id = wslam::LandmarkId{id.v}, .position_world = position});
    }
    for (const auto& [id, factor] : shared_.smart_factors) {
        // A factor submitted after the last drain observes a pose the estimate
        // doesn't have yet
        const bool all_keys_present = std::ranges::all_of(
            factor->keys(),
            [&](const gtsam::Key key) { return latest_values_.exists(key); });
        if (!all_keys_present) {
            continue;
        }
        const gtsam::TriangulationResult triangulation
            = factor->point(latest_values_);
        if (triangulation.has_value()) {
            snapshot.landmarks.push_back(
                LandmarkEstimate{.id = wslam::LandmarkId{id.v},
                                 .position_world = triangulation.value()});
        }
    }

    snapshot.stats.keyframes = snapshot.keyframes.size();
    snapshot.stats.landmarks = snapshot.landmarks.size();
    snapshot.stats.factors = last_factor_count_;
    snapshot.gravity_world = shared_.gravity_world;

    WSLAM_ASSERT(snapshot.stats.keyframes <= shared_.keyframes_processed,
                 "snapshot has {} keyframes but only {} were accepted",
                 snapshot.stats.keyframes, shared_.keyframes_processed);

    storage_.set(ResourceIdentifier::MapSnapshotName, std::move(snapshot));
}

std::optional<std::string> SubmitSmootherPass::flush() {
    // No later keyframe drains the last update
    if (auto err = drainPending()) {
        return err;
    }

    spdlog::info(LOG_ID
                 " flush(): publishing final snapshot ({} archived keyframes, "
                 "{} archived landmarks)",
                 shared_.archived_poses.size(),
                 shared_.archived_landmarks.size());
    publishSnapshot();
    return std::nullopt;
}

std::string DrainSmootherPass::getId() const { return "[Drain Smoother pass]"; }

std::optional<std::string> DrainSmootherPass::initialize() {
    return std::nullopt;
}

std::optional<std::string> DrainSmootherPass::execute() {
    spdlog::info("[Drain Smoother pass] Executing");
    return target_.drainPending();
}
