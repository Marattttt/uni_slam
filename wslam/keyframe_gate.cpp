#include "keyframe_gate.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <flat_set>
#include <numbers>
#include <utility>
#include <vector>

#include "common.hpp"
#include "models.hpp"
#include "provider_base.hpp"

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

// Inverse OpenCV radial-tangential distortion via fixed-point iteration.
// Converts an observed (distorted) normalised image-plane coordinate
// back to its ideal (undistorted) normalised coordinate. Mirrors the
// helper in triangulate_cpu.cpp; duplicated to avoid a public header
// dependency.
Eigen::Vector2d UndistortNormalised(const Eigen::Vector2d& dist,
                                    const Eigen::Vector4d& d) {
    constexpr uint32_t kMaxIter = 8;
    constexpr double kTol = 1e-9;
    Eigen::Vector2d xy = dist;
    for (uint32_t i = 0; i < kMaxIter; ++i) {
        const double x = xy.x();
        const double y = xy.y();
        const double r2 = x * x + y * y;
        const double radial = 1.0 + d[0] * r2 + d[1] * r2 * r2;
        const double dx_t = 2.0 * d[2] * x * y + d[3] * (r2 + 2.0 * x * x);
        const double dy_t = d[2] * (r2 + 2.0 * y * y) + 2.0 * d[3] * x * y;
        const Eigen::Vector2d next((dist.x() - dx_t) / radial,
                                   (dist.y() - dy_t) / radial);
        if ((next - xy).norm() < kTol) {
            xy = next;
            break;
        }
        xy = next;
    }
    return xy;
}

// Convert a feature's LOD-0 distorted pixel to its undistorted LOD-0
// pixel using camera intrinsics + radtan distortion coefficients.
Eigen::Vector2d UndistortLod0Pixel(const Feature& f,
                                   const Eigen::Vector4d& intrinsics,
                                   const Eigen::Vector4d& distortion) {
    const Eigen::Vector2d px = ToLod0Pixel(f);
    const double fu = intrinsics[0];
    const double fv = intrinsics[1];
    const double cu = intrinsics[2];
    const double cv = intrinsics[3];
    const Eigen::Vector2d xy_dist((px.x() - cu) / fu, (px.y() - cv) / fv);
    const Eigen::Vector2d xy = UndistortNormalised(xy_dist, distortion);
    return {fu * xy.x() + cu, fv * xy.y() + cv};
}

// Mean accel reading (in IMU frame) across a window of samples. Used at
// startup to estimate gravity-in-body before the IMU starts moving; the
// caller is responsible for ensuring the window is taken from an
// approximately stationary segment.
Eigen::Vector3d MeanAccelImu(std::span<const data::IMUReading> w) {
    if (w.empty()) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& r : w) {
        sum += Eigen::Vector3d(static_cast<double>(r.ax()),
                               static_cast<double>(r.ay()),
                               static_cast<double>(r.az()));
    }
    return sum / static_cast<double>(w.size());
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

    // Compute the gating signals we need for non-first keyframes in a
    // single pass over the triangulation result: median pixel parallax
    // between the matched feature pair, and the count of landmarks whose
    // `feat_prev` is not yet tracked (i.e. genuinely new map content).
    double median_parallax_px = 0.0;
    size_t new_landmark_count = 0;
    if (state_.has_origin) {
        std::vector<double> parallaxes;
        parallaxes.reserve(tri.landmarks.size());
        for (const auto& lm : tri.landmarks) {
            const auto pp = ToLod0Pixel(lm.feat_prev);
            const auto pc = ToLod0Pixel(lm.feat_curr);
            parallaxes.push_back((pc - pp).norm());
            if (!state_.active_landmarks.contains(lm.feat_prev)) {
                ++new_landmark_count;
            }
        }
        const auto mid = parallaxes.begin()
                         + static_cast<std::ptrdiff_t>(parallaxes.size() / 2);
        std::nth_element(parallaxes.begin(), mid, parallaxes.end());
        median_parallax_px = *mid;

        // Motion gate: accept only if EITHER the recovered rotation OR the
        // median pixel parallax is large enough. The OR is essential —
        // pure-translation frames have ~zero rotation but real parallax.
        const bool tiny_motion = (rot_angle < opts_.min_rotation_rad)
                                  && (median_parallax_px < opts_.min_parallax_px);
        if (tiny_motion) {
            spdlog::debug(LOG_ID
                          " Tiny motion: rot={:.4f} rad (< {:.4f}) AND "
                          "median parallax={:.2f} px (< {:.2f}); skipping",
                          rot_angle, opts_.min_rotation_rad,
                          median_parallax_px, opts_.min_parallax_px);
            storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
            return std::nullopt;
        }

        // Novelty gate: a keyframe that adds no new landmarks just piles
        // re-observation factors onto existing tracks. Skip it.
        if (new_landmark_count < opts_.min_new_landmarks) {
            spdlog::debug(LOG_ID
                          " Only {} new landmarks (< {} required); skipping",
                          new_landmark_count, opts_.min_new_landmarks);
            storage.set(ResourceIdentifier::MapDeltaName, std::move(delta));
            return std::nullopt;
        }
    }

    // Allocate this frame's pose key.
    delta.accepted = true;
    delta.pose_id = PoseId{state_.next_pose_id++};

    // World pose. We anchor the first accepted keyframe at the world origin
    // and chain every subsequent one off the previous keyframe's
    // *predicted* (initial-guess) pose taken from `predicted_values`, not
    // off `latest_values`. iSAM2 runs on a worker thread, so at this point
    // the previous keyframe's optimised pose is typically still in flight
    // and hasn't been written back to `latest_values` yet. Chaining off
    // predicted_values keeps the front-end's pose chain deterministic and
    // independent of back-end completion timing.
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
        if (!state_.predicted_values.exists(prev_key)) {
            return std::format(
                "keyframe gate: predicted_values has no entry for prev pose "
                "id={}",
                prev.v);
        }
        const auto& prev_pose
            = state_.predicted_values.at<gtsam::Pose3>(prev_key);
        const Eigen::Matrix3d r_wp = prev_pose.rotation().matrix();
        const Eigen::Vector3d t_wp = prev_pose.translation();
        const Eigen::Matrix3d r_wc = r_wp * tri.R_prev_to_curr.transpose();
        const Eigen::Vector3d t_wc = t_wp - r_wc * tri.t_prev_to_curr;
        delta.R_world_cam = r_wc;
        delta.t_world_cam = t_wc;
        delta.R_rel = tri.R_prev_to_curr;
        delta.t_rel = tri.t_prev_to_curr;
    }

    // Record the new keyframe's initial pose in predicted_values so the
    // next frame can chain off it without waiting for iSAM.
    state_.predicted_values.insert(
        MappingState::poseKey(delta.pose_id),
        gtsam::Pose3(gtsam::Rot3(delta.R_world_cam),
                     gtsam::Point3(delta.t_world_cam)));

    // IMU windowing: copy the samples that fall in (prev_kf_ts, curr_kf_ts]
    // into the delta, then trim the AnyBag buffer so the next gate doesn't
    // double-count. The current frame's timestamp is published by the
    // sensor loader at the top of every pipeline iteration.
    {
        const auto ts_ptr = storage.getPtr<uint64_t>(
            ResourceIdentifier::FrameTimestampNsName);
        if (!ts_ptr.has_value()) {
            return "keyframe gate: missing current-frame timestamp under '"
                   + ResourceIdentifier::FrameTimestampNsName + "'";
        }
        const uint64_t curr_ts = **ts_ptr;
        delta.curr_kf_ts_ns = curr_ts;
        delta.prev_kf_ts_ns = state_.last_keyframe_ts_ns.value_or(curr_ts);

        // Take the IMU buffer by value so we can replace it with the
        // post-window tail in one shot below.
        auto imu_buf_opt = storage.get<std::vector<data::IMUReading>>(
            ResourceIdentifier::GetImuVecName());
        if (imu_buf_opt.has_value()) {
            auto& imu_buf = imu_buf_opt.value();

            // On the very first accepted keyframe, take a stationary-window
            // average across the IMU samples we have so far to estimate
            // gravity in the *body* (=IMU) frame, then rotate into the
            // camera/world frame for downstream preintegration. Whether the
            // dataset start is truly stationary is the caller's problem;
            // EuRoC sequences typically start that way.
            if (delta.is_first_keyframe && !imu_buf.empty()) {
                const auto cam_ptr
                    = storage.getPtr<data::CamSensorParams>(
                        ResourceIdentifier::GetCameraIntrinsicsName(0));
                if (cam_ptr.has_value()) {
                    // T_body_cam.rotation(); world == first cam frame, so
                    // gravity_in_world = R_cam_body * gravity_in_body.
                    const Eigen::Matrix3d r_cam_body
                        = (**cam_ptr).T_BS.block<3, 3>(0, 0).transpose();
                    const Eigen::Vector3d g_body
                        = MeanAccelImu(std::span<const data::IMUReading>(
                            imu_buf.data(), imu_buf.size()));
                    // The accel reading on a stationary IMU is the negative
                    // of gravitational acceleration (specific force points
                    // opposite to free-fall). Negate to recover the actual
                    // gravity vector.
                    state_.gravity_world = -r_cam_body * g_body;
                    state_.gravity_initialised = true;
                    spdlog::info(LOG_ID
                                 " Initialised gravity in world from {} "
                                 "stationary IMU samples: g_world="
                                 "({:.3f},{:.3f},{:.3f}) "
                                 "|g|={:.3f} m/s^2",
                                 imu_buf.size(), state_.gravity_world.x(),
                                 state_.gravity_world.y(),
                                 state_.gravity_world.z(),
                                 state_.gravity_world.norm());
                } else {
                    spdlog::warn(LOG_ID
                                 " No camera params under '{}'; cannot "
                                 "initialise gravity",
                                 ResourceIdentifier::GetCameraIntrinsicsName(0));
                }
            }

            // Slice out everything with ts in (prev_ts, curr_ts]. On the
            // first keyframe we take an empty slice (no prev to integrate
            // from) and drop everything older than curr_ts.
            const uint64_t lo = delta.prev_kf_ts_ns;
            const uint64_t hi = curr_ts;
            if (!delta.is_first_keyframe) {
                delta.imu_between.reserve(imu_buf.size());
                for (const auto& r : imu_buf) {
                    if (r.timestamp > lo && r.timestamp <= hi) {
                        delta.imu_between.push_back(r);
                    }
                }
            }
            // Keep samples strictly after the current keyframe — they
            // belong to the next integration window.
            std::vector<data::IMUReading> tail;
            tail.reserve(imu_buf.size());
            for (auto& r : imu_buf) {
                if (r.timestamp > hi) {
                    tail.push_back(r);
                }
            }
            storage.set(ResourceIdentifier::GetImuVecName(), std::move(tail));
        }
        state_.last_keyframe_ts_ns = curr_ts;
    }

    // Camera intrinsics + distortion are needed to pre-undistort
    // observations before they're added to smart factors downstream
    // (which use a pinhole Cal3_S2). Required: triangulation already
    // depends on the same intrinsics being in AnyBag, so a missing entry
    // would mean upstream failed.
    const auto cam_for_undistort = storage.getPtr<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    if (!cam_for_undistort.has_value()) {
        return "keyframe gate: missing camera intrinsics under '"
               + ResourceIdentifier::GetCameraIntrinsicsName(0) + "'";
    }
    const auto& intrinsics_vec = (**cam_for_undistort).intrinsics;
    const auto& distortion_vec
        = (**cam_for_undistort).distortion_coefficients;
    const auto Undistort = [&](const Feature& f) {
        return UndistortLod0Pixel(f, intrinsics_vec, distortion_vec);
    };

    // Landmark association. Build the next active_landmarks map fresh: a
    // landmark observed *this* frame stays alive only if it's actually seen.
    std::flat_map<Feature, LandmarkId> next_active;

    delta.observations.reserve(tri.landmarks.size()
                               + (delta.is_first_keyframe ? tri.landmarks.size()
                                                          : 0));
    delta.new_landmarks_world.reserve(tri.landmarks.size());

    size_t dropped_low_parallax = 0;
    for (const auto& lm : tri.landmarks) {
        const auto it = state_.active_landmarks.find(lm.feat_prev);
        LandmarkId id{};
        bool is_new = false;
        if (it != state_.active_landmarks.end()) {
            id = it->second;
        } else {
            // Per-landmark parallax gate: a new landmark whose feature
            // displacement between prev and curr keyframe is below
            // threshold can't be reliably triangulated and the resulting
            // marginal Hessian is near-singular along the depth axis.
            // Drop it rather than feed iSAM2 a degenerate variable. The
            // feature may be re-acquired on a later keyframe where the
            // baseline is bigger.
            const auto pp = ToLod0Pixel(lm.feat_prev);
            const auto pc = ToLod0Pixel(lm.feat_curr);
            if ((pc - pp).norm() < opts_.min_landmark_parallax_px) {
                ++dropped_low_parallax;
                continue;
            }
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
            // With smart factors, landmarks are *not* graph variables —
            // they're triangulated internally by SmartProjectionPoseFactor
            // at each iSAM linearisation. So we don't insert into
            // predicted_values for them.

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
                    .pixel_lod0 = Undistort(lm.feat_prev),
                });
            }
        }
        delta.observations.push_back(LandmarkObservation{
            .pose = delta.pose_id,
            .landmark = id,
            .pixel_lod0 = Undistort(lm.feat_curr),
        });
    }

    spdlog::info(LOG_ID
                 " Accepted keyframe pose_id={} (first={}, prev={}), "
                 "obs={}, new_landmarks={}, dropped_low_parallax={}",
                 delta.pose_id.v, delta.is_first_keyframe,
                 delta.prev_pose_id.has_value()
                     ? std::to_string(delta.prev_pose_id.value().v)
                     : std::string{"-"},
                 delta.observations.size(), delta.new_landmarks_world.size(),
                 dropped_low_parallax);

#ifndef NDEBUG
    // Invariant: every observation references a landmark that either was
    // just declared this frame or was created on a previous keyframe.
    // The active_landmarks map is updated below to contain the latter
    // set, so checking against the union of (just-declared, prior tracks)
    // covers it.
    {
        std::flat_set<LandmarkId> known;
        for (const auto& [id, _] : delta.new_landmarks_world) {
            known.insert(id);
        }
        for (const auto& [_feat, id] : state_.active_landmarks) {
            known.insert(id);
        }
        for (const auto& obs : delta.observations) {
            assert(known.contains(obs.landmark));
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
