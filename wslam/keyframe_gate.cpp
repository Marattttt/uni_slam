#include "keyframe_gate.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <flat_set>
#include <numbers>
#include <utility>

#include "common.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[Keyframe Gate pass]"

KeyframeGatePass::KeyframeGatePass(MappingState& state,
                                   std::shared_ptr<compute::GPU> gpu, Opts opts)
    : compute::Pass(std::move(gpu)), state_(state), opts_(opts) {}

KeyframeGatePass::KeyframeGatePass(MappingState& state,
                                   std::shared_ptr<compute::GPU> gpu)
    : KeyframeGatePass(state, std::move(gpu), Opts{}) {}

std::string KeyframeGatePass::getId() const { return LOG_ID; }

std::optional<std::string> KeyframeGatePass::initialize() {
    spdlog::info(LOG_ID
                 " Initializing (min_landmarks={}, min_rotation_rad={:.4f})",
                 opts_.min_landmarks, opts_.min_rotation_rad);
    if (storage_ == nullptr) {
        return "keyframe gate: storage not set before initialize()";
    }
    return std::nullopt;
}

namespace {

constexpr double LodScale(uint32_t lod) {
    double scale = 1.0;
    for (uint32_t i = 0; i < lod; ++i) {
        scale *= GPUConst::lod_scale_factor;
    }
    return scale;
}

constexpr Eigen::Vector2d ToLod0Pixel(const Feature& f) {
    const double scale = LodScale(f.lod);
    return {static_cast<double>(f.x) * scale,
            static_cast<double>(f.y) * scale};
}

// Returns the rotation angle (radians) of a 3x3 rotation, clamped for
// numerical safety. Mirrors the helper in triangulate_cpu.cpp.
double RotationAngle(const Eigen::Matrix3d& r) {
    const double trace = std::clamp(r.trace(), -1.0, 3.0);
    return std::acos(std::clamp(0.5 * (trace - 1.0), -1.0, 1.0));
}

}  // namespace

std::optional<std::string> KeyframeGatePass::execute() {
    spdlog::info(LOG_ID " Executing");

    assert(storage_ != nullptr);
    auto& storage = *storage_;

    MapDelta delta;

    const auto tri_ptr = storage.getPtr<TriangulationResult>(
        ResourceIdentifier::TriangulationResultName);
    if (!tri_ptr.has_value()) {
        spdlog::warn(LOG_ID " No TriangulationResult; emitting empty MapDelta");
        storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
        return std::nullopt;
    }
    const auto& tri = *tri_ptr.value();

    if (!tri.stats.pose_recovered) {
        spdlog::warn(LOG_ID
                     " Triangulation reports no pose; emitting empty MapDelta");
        storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
        return std::nullopt;
    }

    if (tri.landmarks.size() < opts_.min_landmarks) {
        spdlog::warn(LOG_ID
                     " Only {} landmarks (< {} required); skipping frame",
                     tri.landmarks.size(), opts_.min_landmarks);
        storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
        return std::nullopt;
    }

    const double rot_angle = RotationAngle(tri.R_prev_to_curr);
    const bool tiny_motion = !state_.has_origin
                                 ? false
                                 : (rot_angle < opts_.min_rotation_rad);
    if (tiny_motion) {
        spdlog::debug(LOG_ID
                      " Rotation {:.6f} rad below threshold {:.6f}; skipping",
                      rot_angle, opts_.min_rotation_rad);
        storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
        return std::nullopt;
    }

    // Allocate this frame's pose key.
    delta.accepted = true;
    delta.pose_id = PoseId{state_.next_pose_id++};

    // World pose. We anchor the first accepted keyframe at the world origin
    // and chain every subsequent one off the most recent latest_values
    // estimate using the relative pose from triangulation.
    if (!state_.has_origin) {
        delta.is_first_keyframe = true;
        delta.R_world_cam.setIdentity();
        delta.t_world_cam.setZero();
        delta.prev_pose_id.reset();
        delta.R_rel.setIdentity();
        delta.t_rel.setZero();
    } else {
        if (!state_.last_accepted_pose.has_value()) {
            return "keyframe gate: origin set but no last_accepted_pose — "
                   "state inconsistent";
        }
        const auto prev = state_.last_accepted_pose.value();
        delta.prev_pose_id = prev;

        // Convert the relative pose triangulation reported (p_curr =
        // R*p_prev + t in *camera* frames) into the world pose chain:
        //   T_w_curr = T_w_prev * T_prev_curr
        // T_prev_curr (cam_prev → cam_curr extrinsic): R^T, -R^T * t when
        // applied to a world pose, i.e. cam_curr-from-world =
        // R * cam_prev-from-world, then translate.
        //
        // Concretely: if T_w_prev = (R_wp, t_wp), then
        //   R_wc = R_wp * R_prev_to_curr^T
        //   t_wc = t_wp - R_wc * t_prev_to_curr
        // because t_prev_to_curr is expressed in the *previous* camera frame.
        const auto prev_key = MappingState::poseKey(prev);
        if (!state_.latest_values.exists(prev_key)) {
            return std::format(
                "keyframe gate: latest_values has no entry for prev pose "
                "id={}",
                prev.v);
        }
        const auto& prev_pose
            = state_.latest_values.at<gtsam::Pose3>(prev_key);
        const Eigen::Matrix3d r_wp = prev_pose.rotation().matrix();
        const Eigen::Vector3d t_wp = prev_pose.translation();
        const Eigen::Matrix3d r_wc = r_wp * tri.R_prev_to_curr.transpose();
        const Eigen::Vector3d t_wc = t_wp - r_wc * tri.t_prev_to_curr;
        delta.R_world_cam = r_wc;
        delta.t_world_cam = t_wc;
        delta.R_rel = tri.R_prev_to_curr;
        delta.t_rel = tri.t_prev_to_curr;
    }

    // Landmark association. Build the next active_landmarks map fresh: a
    // landmark observed *this* frame stays alive only if it's actually seen.
    std::flat_map<Feature, LandmarkId> next_active;

    delta.observations.reserve(tri.landmarks.size()
                               + (delta.is_first_keyframe ? tri.landmarks.size()
                                                          : 0));
    delta.new_landmarks_world.reserve(tri.landmarks.size());

    for (const auto& lm : tri.landmarks) {
        const auto it = state_.active_landmarks.find(lm.feat_prev);
        LandmarkId id{};
        bool is_new = false;
        if (it != state_.active_landmarks.end()) {
            id = it->second;
        } else {
            id = LandmarkId{state_.next_landmark_id++};
            is_new = true;
        }
        next_active.insert_or_assign(lm.feat_curr, id);

        if (is_new) {
            // Initial world position: transform the landmark from current
            // camera coordinates to world.
            const Eigen::Vector3d p_world
                = delta.R_world_cam * lm.position_cam_curr + delta.t_world_cam;
            delta.new_landmarks_world.emplace_back(id, p_world);

            // For new landmarks introduced after the first keyframe we also
            // emit a feat_prev observation at the *previous* keyframe pose.
            // Two views in the graph constrain the landmark's depth and keep
            // the iSAM2 linear system well-posed without leaning on a tight
            // position prior. On the first keyframe there is no prev pose in
            // the graph, so the factor builder anchors first-keyframe
            // landmarks with their triangulated positions instead.
            if (delta.prev_pose_id.has_value()) {
                delta.observations.push_back(LandmarkObservation{
                    .pose = delta.prev_pose_id.value(),
                    .landmark = id,
                    .pixel_lod0 = ToLod0Pixel(lm.feat_prev),
                });
            }
        }
        delta.observations.push_back(LandmarkObservation{
            .pose = delta.pose_id,
            .landmark = id,
            .pixel_lod0 = ToLod0Pixel(lm.feat_curr),
        });
    }

    spdlog::info(LOG_ID
                 " Accepted keyframe pose_id={} (first={}, prev={}), "
                 "obs={}, new_landmarks={}",
                 delta.pose_id.v, delta.is_first_keyframe,
                 delta.prev_pose_id.has_value()
                     ? std::to_string(delta.prev_pose_id.value().v)
                     : std::string{"-"},
                 delta.observations.size(), delta.new_landmarks_world.size());

#ifndef NDEBUG
    // Invariant: observations always reference variables we either just
    // declared (new) or that are already in latest_values from a prior frame.
    {
        std::flat_set<LandmarkId> known;
        for (const auto& [id, _] : delta.new_landmarks_world) {
            known.insert(id);
        }
        for (const auto& obs : delta.observations) {
            assert(known.contains(obs.landmark)
                   || state_.latest_values.exists(
                          MappingState::landmarkKey(obs.landmark)));
        }
    }
#endif

    state_.active_landmarks = std::move(next_active);
    state_.last_accepted_pose = delta.pose_id;
    if (!state_.has_origin) {
        state_.has_origin = true;
    }

    storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
    return std::nullopt;
}
