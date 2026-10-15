#include "feed_loop_closure.hpp"

#include <spdlog/spdlog.h>

#include <utility>

#include "common.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;

#define LOG_ID "[Feed Loop Closure pass]"

std::string FeedLoopClosurePass::getId() const { return LOG_ID; }

std::optional<std::string> FeedLoopClosurePass::initialize() {
    spdlog::info(LOG_ID " Initializing (no-op)");
    return std::nullopt;
}

std::optional<std::string> FeedLoopClosurePass::execute() {
    spdlog::info(LOG_ID " Executing");

    // The previous keyframe's drain ran in this stage, before the new one
    // was submitted
    pushMapUpdate();

    const auto frame
        = storage_.getPtr<data::FrameBW>(ResourceIdentifier::GetFrameName(0));
    if (!frame) {
        return "no frame for the keyframe image";
    }
    const auto timestamp
        = storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName);
    if (!timestamp) {
        return "no frame timestamp";
    }

    auto features
        = storage_
              .get<std::shared_ptr<const v2::raco_lightglue::FrameFeatures>>(
                  ResourceIdentifier::OrtFrameFeaturesName);
    if (!features || features.value() == nullptr) {
        return "no model features for the keyframe";
    }

    loop_closure::KeyframeImageEvent event{
        .keyframe = shared_.last_pose.v,
        .timestamp_ns = timestamp.value(),
        .image = loop_closure::PlaceImage::FromFrame(*frame.value()),
        .features = std::move(features).value(),
        .landmarks = {},
    };
    // Tracks are keyed by the keyframe's keypoint index, which indexes the
    // features
    event.landmarks.reserve(shared_.active_landmarks.size());
    for (const auto& [index, tracked] : shared_.active_landmarks) {
        event.landmarks.push_back(loop_closure::KeyframeLandmark{
            .landmark = tracked.landmark.v,
            .keypoint_index = index,
        });
    }
    worker_->push(std::move(event));

    pullCorrection();
    return std::nullopt;
}

void FeedLoopClosurePass::flush() {
    pushMapUpdate();
    pullCorrection();
}

void FeedLoopClosurePass::pushMapUpdate() {
    loop_closure::MapUpdateEvent update;

    if (!gravity_sent_ && shared_.keyframes_processed > 0) {
        update.gravity_world = shared_.gravity_world;
        gravity_sent_ = true;
    }

    const PoseId drained = shared_.last_drained_pose;
    if (drained.v > last_sent_drained_.v) {
        // Drains settle one keyframe each, in order
        const gtsam::Key key = Pose(drained);
        if (shared_.predicted_values.exists(key)) {
            update.drained_poses.emplace_back(
                drained.v, shared_.predicted_values.at<gtsam::Pose3>(key));
        } else if (const auto archived = shared_.archived_poses.find(drained);
                   archived != shared_.archived_poses.end()) {
            update.drained_poses.emplace_back(drained.v, archived->second);
        }
        last_sent_drained_ = drained;
    }

    for (const PoseId& id : shared_.poses_archived_since_feed) {
        update.archived_poses.emplace_back(id.v, shared_.archived_poses.at(id));
    }
    for (const LandmarkId& id : shared_.landmarks_archived_since_feed) {
        update.archived_landmarks.emplace_back(
            id.v, shared_.archived_landmarks.at(id));
    }
    shared_.poses_archived_since_feed.clear();
    shared_.landmarks_archived_since_feed.clear();

    if (update.drained_poses.empty() && update.archived_poses.empty()
        && update.archived_landmarks.empty() && !update.gravity_world) {
        return;
    }
    worker_->push(std::move(update));
}

void FeedLoopClosurePass::pullCorrection() {
    const auto correction
        = worker_->latestCorrection(shared_.drift_correction_version);
    if (!correction) {
        return;
    }

    shared_.drift_correction = correction->T_drift;
    shared_.drift_correction_version = correction->version;

    const gtsam::Pose3& drift = correction->T_drift;
    spdlog::info(LOG_ID
                 " drift correction v{} at x{} after {} loops: {:.3f} m, "
                 "{:.2f} deg yaw",
                 correction->version, correction->anchor_keyframe,
                 correction->loops, drift.translation().norm(),
                 drift.rotation().axisAngle().second * 180.0 / M_PI);
}
