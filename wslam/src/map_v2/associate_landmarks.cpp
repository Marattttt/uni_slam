#include "associate_landmarks.hpp"

#include <spdlog/spdlog.h>

#include <flat_map>
#include <flat_set>
#include <utility>

#include "common.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;

#define LOG_ID "[Associate Landmarks pass]"

std::string AssociateLandmarksPass::getId() const { return LOG_ID; }

std::optional<std::string> AssociateLandmarksPass::initialize() {
    spdlog::info(LOG_ID " Initializing (no-op)");
    return std::nullopt;
}

std::optional<std::string> AssociateLandmarksPass::execute() {
    spdlog::info(LOG_ID " Executing");

    auto candidate = storage_.take<KeyframeCandidate>(
        ResourceIdentifier::KeyframeCandidateName);
    if (!candidate) {
        return "no keyframe candidate";
    }

    MapChanges changes{
        .pose_id = candidate->pose_id,
        .prev_pose_id = candidate->prev_pose_id,
        .prev_kf_ts = candidate->prev_kf_ts,
        .curr_kf_ts = candidate->curr_kf_ts,
        .R_world_cam = candidate->R_world_cam,
        .t_world_cam = candidate->t_world_cam,
        .R_rel = candidate->R_rel,
        .t_rel = candidate->t_rel,
        .observations = {},
        .imu = std::move(candidate->imu),
    };
    changes.observations.reserve(candidate->matches.size() * 2);

    std::flat_map<uint32_t, TrackedKeypoint> next_active;
    std::flat_set<uint32_t> consumed_prev;

    size_t tracked = 0;
    size_t fresh = 0;
    size_t marginalised = 0;
    size_t pixel_mismatches = 0;

    for (const CandidateMatch& match : candidate->matches) {
        // The model matches one to one; guard against a reference keypoint
        // observing two landmarks anyway
        if (!consumed_prev.insert(match.prev_index).second) {
            continue;
        }

        const TrackLookupResult lookup
            = LookupTrack(shared_, match.prev_index, match.prev_pixel_observed,
                          opts_.association_tolerance_px);

        if (lookup.outcome == TrackLookup::PixelMismatch) {
            ++pixel_mismatches;
        }

        const bool was_tracked = lookup.outcome == TrackLookup::Tracked;
        const bool in_window
            = was_tracked && shared_.smart_factors.contains(lookup.landmark);
        if (was_tracked && !in_window) {
            ++marginalised;
        }

        LandmarkId landmark{0};
        if (in_window) {
            landmark = lookup.landmark;
            ++tracked;
        } else {
            shared_.last_landmark.v++;
            landmark = shared_.last_landmark;
            shared_.landmark_anchor.emplace(landmark, changes.pose_id);
            ++fresh;

            // A marginalised landmark's factor already held the previous
            // keyframe's observation, now part of the marginal: its successor
            // starts from the current keyframe only
            if (!changes.isFirstKeyframe() && !was_tracked) {
                changes.observations.push_back(Observation{
                    .pose = changes.prev_pose_id,
                    .landmark = landmark,
                    .pixel = match.prev_pixel,
                });
            }
        }

        changes.observations.push_back(Observation{
            .pose = changes.pose_id,
            .landmark = landmark,
            .pixel = match.curr_pixel,
        });

        next_active.insert_or_assign(
            match.curr_index,
            TrackedKeypoint{.landmark = landmark,
                            .pixel = match.curr_pixel_observed});
    }

    spdlog::info(LOG_ID
                 " x{}: {} tracked, {} new, {} continued tracks already "
                 "marginalised, {} keypoint pixel mismatches",
                 changes.pose_id.v, tracked, fresh, marginalised,
                 pixel_mismatches);

    shared_.active_landmarks = std::move(next_active);
    storage_.set(ResourceIdentifier::MapDeltaName, std::move(changes));

    return std::nullopt;
}
