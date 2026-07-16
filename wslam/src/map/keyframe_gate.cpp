#include "keyframe_gate.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <flat_map>
#include <flat_set>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "common.hpp"
#include "models.hpp"
#include "data/provider_base.hpp"

using namespace wslam;

#define LOG_ID "[Keyframe Gate pass]"

KeyframeGatePass::KeyframeGatePass(std::shared_ptr<MappingState> state,
                                   Opts opts)
    : state_(std::move(state)), opts_(opts) {}

KeyframeGatePass::KeyframeGatePass(std::shared_ptr<MappingState> state)
    : KeyframeGatePass(std::move(state), Opts{}) {}

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
    return {static_cast<double>(f.x) * scale, static_cast<double>(f.y) * scale};
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

// Mean gyro reading across a window. On a stationary platform the true
// angular rate is zero, so this mean *is* the gyro bias.
Eigen::Vector3d MeanGyroImu(std::span<const data::IMUReading> w) {
    if (w.empty()) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& r : w) {
        sum += Eigen::Vector3d(static_cast<double>(r.wx()),
                               static_cast<double>(r.wy()),
                               static_cast<double>(r.wz()));
    }
    return sum / static_cast<double>(w.size());
}

// Quietest sub-span of `window_size` samples in the buffer, scored by
// std(|accel|) + mean |gyro - window_mean_gyro|. Both terms are immune
// to the gyro *bias* (EuRoC's is ~0.08 rad/s, far above any rotation
// threshold one could set) and only mildly sensitive to prop vibration,
// which is exactly the regime a parked MAV sits in. Picking the minimum
// rather than the first window below a threshold makes the scan
// dataset-agnostic; `found` only reports whether even the best window
// looks plausibly stationary. Buffer too small ⇒ whole buffer,
// recovering the old "average everything" behaviour.
struct StationaryWindow {
    std::span<const data::IMUReading> samples;
    double score = std::numeric_limits<double>::infinity();
    bool found = false;
};

StationaryWindow FindStationaryWindow(std::span<const data::IMUReading> buf,
                                      size_t window_size, double max_score) {
    if (window_size == 0 || buf.size() < window_size) {
        return {.samples = buf};
    }
    StationaryWindow best{.samples = buf};
    for (size_t begin = 0; begin + window_size <= buf.size();
         begin += std::max<size_t>(1, window_size / 2)) {
        const auto window = buf.subspan(begin, window_size);
        Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
        double accel_mag_sum = 0.0;
        double accel_mag_sq_sum = 0.0;
        for (const auto& r : window) {
            gyro_mean += Eigen::Vector3d(static_cast<double>(r.wx()),
                                         static_cast<double>(r.wy()),
                                         static_cast<double>(r.wz()));
            const double accel_mag
                = Eigen::Vector3d(static_cast<double>(r.ax()),
                                  static_cast<double>(r.ay()),
                                  static_cast<double>(r.az()))
                      .norm();
            accel_mag_sum += accel_mag;
            accel_mag_sq_sum += accel_mag * accel_mag;
        }
        const auto n = static_cast<double>(window_size);
        gyro_mean /= n;
        double gyro_dev_sum = 0.0;
        for (const auto& r : window) {
            gyro_dev_sum += (Eigen::Vector3d(static_cast<double>(r.wx()),
                                             static_cast<double>(r.wy()),
                                             static_cast<double>(r.wz()))
                             - gyro_mean)
                                .norm();
        }
        const double accel_mean = accel_mag_sum / n;
        const double accel_var
            = std::max(0.0, accel_mag_sq_sum / n - accel_mean * accel_mean);
        const double score = std::sqrt(accel_var) + gyro_dev_sum / n;
        if (score < best.score) {
            best = StationaryWindow{
                .samples = window, .score = score, .found = score < max_score};
        }
    }
    return best;
}

}  // namespace

std::optional<std::string> KeyframeGatePass::execute() {
    spdlog::info(LOG_ID " Executing");

    assert(storage_ != nullptr);
    auto& storage = *storage_;

    MapDelta delta;

    // Every exit funnels through here. Besides publishing the MapDelta,
    // it tells LoadDataCPUPass whether to advance the reference feature
    // set (FeatureSet(1)) next frame: on acceptance the reference must
    // become the just-accepted keyframe's features (so future matching /
    // triangulation spans exactly the new graph edge), and pre-origin it
    // advances every frame (frame-to-frame matching keeps the bootstrap
    // baseline short, putting the first keyframe at motion onset).
    const bool had_origin = state_->has_origin;
    const auto finish = [&](MapDelta d) -> std::optional<std::string> {
        if (d.accepted || !had_origin) {
            storage.set(ResourceIdentifier::FeatureReferenceAdvanceName, true);
        }
        storage.set(ResourceIdentifier::MapDeltaName, std::move(d));
        return std::nullopt;
    };

    const auto tri_ptr = storage.getPtr<TriangulationResult>(
        ResourceIdentifier::TriangulationResultName);
    if (!tri_ptr.has_value()) {
        spdlog::warn(LOG_ID " No TriangulationResult; emitting empty MapDelta");
        return finish(std::move(delta));
    }
    const auto& tri = *tri_ptr.value();

    if (!tri.stats.pose_recovered) {
        spdlog::warn(LOG_ID
                     " Triangulation reports no pose; emitting empty MapDelta");
        return finish(std::move(delta));
    }

    if (tri.landmarks.size() < opts_.min_landmarks) {
        spdlog::warn(LOG_ID
                     " Only {} landmarks (< {} required); skipping frame",
                     tri.landmarks.size(), opts_.min_landmarks);
        return finish(std::move(delta));
    }

    const double rot_angle = RotationAngle(tri.rotation);

    // Association candidates: ALL RANSAC inliers, not just the subset
    // that survived triangulation's cheirality/reprojection filters.
    // With smart projection factors the landmark is never a graph
    // variable, so a pair doesn't need a pre-validated 3D point to be
    // useful — the epipolar (Sampson) gate is enough, and the smart
    // factor re-triangulates internally at every linearisation anyway
    // (ZERO_ON_DEGENERACY + dynamic outlier rejection self-clean the
    // rest). Associating on the triangulated subset capped the
    // re-observation rate at ~10%: each keyframe's triangulated
    // landmarks are a thin sample of its inliers, so consecutive
    // keyframes rarely picked the same feature twice.
    const auto ransac_ptr
        = storage.getPtr<RansacResult>(ResourceIdentifier::RansacResultName);
    if (!ransac_ptr.has_value()) {
        return "keyframe gate: missing RansacResult under '"
               + ResourceIdentifier::RansacResultName + "'";
    }
    std::vector<FeaturePair> pairs;  // (feat_curr, feat_prev)
    {
        const auto& inliers = (**ransac_ptr).inliers;
        pairs.reserve(std::ranges::fold_left(
            inliers, 0UZ,
            [](size_t acc, const auto& m) { return acc + m.size(); }));
        for (const auto& lod_map : inliers) {
            for (const auto& [curr, prev] : lod_map) {
                pairs.emplace_back(curr, prev);
            }
        }
    }
    if (pairs.empty()) {
        spdlog::warn(LOG_ID " RANSAC reports no inlier pairs; skipping");
        return finish(std::move(delta));
    }

    // Per-pair pixel parallax (kept index-aligned with `pairs` — the
    // association loop below reuses it), plus the two gating signals:
    // median parallax and the count of pairs that would genuinely enter
    // the map as NEW content (untracked feat_prev AND enough parallax
    // to survive the per-landmark filter below).
    std::vector<double> parallaxes;
    parallaxes.reserve(pairs.size());
    size_t new_landmark_count = 0;
    for (const auto& [feat_curr, feat_prev] : pairs) {
        const auto pp = ToLod0Pixel(feat_prev);
        const auto pc = ToLod0Pixel(feat_curr);
        const double parallax = (pc - pp).norm();
        parallaxes.push_back(parallax);
        if (!state_->active_landmarks.contains(feat_prev)
            && parallax >= opts_.min_landmark_parallax_px) {
            ++new_landmark_count;
        }
    }
    // Median on a scratch copy — nth_element reorders, and `parallaxes`
    // must stay index-aligned with `pairs`.
    std::vector<double> parallaxes_scratch = parallaxes;
    const auto mid
        = parallaxes_scratch.begin()
          + static_cast<std::ptrdiff_t>(parallaxes_scratch.size() / 2);
    std::nth_element(parallaxes_scratch.begin(), mid, parallaxes_scratch.end());
    const double median_parallax_px = *mid;

    if (!state_->has_origin) {
        // Bootstrap gate: don't anchor the gauge while the platform is
        // still sitting on the ground. Matching is frame-to-frame here,
        // so even a small median parallax means motion has started.
        if (median_parallax_px < opts_.bootstrap_parallax_px) {
            spdlog::debug(LOG_ID
                          " Bootstrap: median parallax={:.2f} px (< {:.2f}); "
                          "still stationary, not anchoring yet",
                          median_parallax_px, opts_.bootstrap_parallax_px);
            return finish(std::move(delta));
        }
        spdlog::info(LOG_ID
                     " Bootstrap: motion onset detected (median "
                     "parallax={:.2f} px, rot={:.4f} rad); anchoring first "
                     "keyframe",
                     median_parallax_px, rot_angle);
    } else {
        // Motion gate: accept only if EITHER the recovered rotation OR the
        // median pixel parallax is large enough. The OR is essential —
        // pure-translation frames have ~zero rotation but real parallax.
        const bool tiny_motion
            = (rot_angle < opts_.min_rotation_rad)
              && (median_parallax_px < opts_.min_parallax_px);
        if (tiny_motion) {
            spdlog::debug(LOG_ID
                          " Tiny motion: rot={:.4f} rad (< {:.4f}) AND "
                          "median parallax={:.2f} px (< {:.2f}); skipping",
                          rot_angle, opts_.min_rotation_rad, median_parallax_px,
                          opts_.min_parallax_px);
            return finish(std::move(delta));
        }

        // Novelty gate: a keyframe that adds no new landmarks just piles
        // re-observation factors onto existing tracks. Skip it.
        if (new_landmark_count < opts_.min_new_landmarks) {
            spdlog::debug(LOG_ID
                          " Only {} new landmarks (< {} required); skipping",
                          new_landmark_count, opts_.min_new_landmarks);
            return finish(std::move(delta));
        }
    }

    // Allocate this frame's pose key.
    delta.accepted = true;
    delta.pose_id = PoseId{state_->next_pose_id++};

    // World pose. We anchor the first accepted keyframe at the world origin
    // and chain every subsequent one off the previous keyframe's
    // *predicted* (initial-guess) pose taken from `predicted_values`, not
    // off `latest_values`. iSAM2 runs on a worker thread, so at this point
    // the previous keyframe's optimised pose is typically still in flight
    // and hasn't been written back to `latest_values` yet. Chaining off
    // predicted_values keeps the front-end's pose chain deterministic and
    // independent of back-end completion timing.
    if (!state_->has_origin) {
        delta.is_first_keyframe = true;
        delta.R_world_cam.setIdentity();
        delta.t_world_cam.setZero();
        delta.prev_pose_id.reset();
        delta.R_rel.setIdentity();
        delta.t_rel.setZero();
    } else {
        if (!state_->last_accepted_pose.has_value()) {
            return "keyframe gate: origin set but no last_accepted_pose — "
                   "state inconsistent";
        }
        const auto prev = state_->last_accepted_pose.value();
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
        if (!state_->predicted_values.exists(prev_key)) {
            return std::format(
                "keyframe gate: predicted_values has no entry for prev pose "
                "id={}",
                prev.v);
        }
        const auto& prev_pose
            = state_->predicted_values.at<gtsam::Pose3>(prev_key);
        const Eigen::Matrix3d r_wp = prev_pose.rotation().matrix();
        const Eigen::Vector3d t_wp = prev_pose.translation();
        const Eigen::Matrix3d r_wc = r_wp * tri.rotation.transpose();
        const Eigen::Vector3d t_wc = t_wp - r_wc * tri.translation;
        delta.R_world_cam = r_wc;
        delta.t_world_cam = t_wc;
        delta.R_rel = tri.rotation;
        delta.t_rel = tri.translation;
    }

    // Record the new keyframe's initial pose in predicted_values so the
    // next frame can chain off it without waiting for iSAM.
    state_->predicted_values.insert(
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
        delta.prev_kf_ts_ns = state_->last_keyframe_ts_ns.value_or(curr_ts);

        // Take the IMU buffer by value so we can replace it with the
        // post-window tail in one shot below.
        auto imu_buf_opt = storage.get<std::vector<data::IMUReading>>(
            ResourceIdentifier::GetImuVecName());
        if (imu_buf_opt.has_value()) {
            auto& imu_buf = imu_buf_opt.value();

            // On the very first accepted keyframe, estimate gravity (and
            // gyro bias) from a *stationary* window of the IMU samples
            // accumulated so far. The buffer spans the dataset start up to
            // motion onset (the bootstrap gate puts the first keyframe
            // there), so a front-to-back scan finds a clean window before
            // the takeoff contaminates the tail. Gravity is estimated in
            // the body (=IMU) frame and rotated into the camera/world
            // frame for downstream preintegration.
            if (delta.is_first_keyframe && !imu_buf.empty()) {
                const auto cam_ptr = storage.getPtr<data::CamSensorParams>(
                    ResourceIdentifier::GetCameraIntrinsicsName(0));
                if (cam_ptr.has_value()) {
                    const auto window = FindStationaryWindow(
                        std::span<const data::IMUReading>(imu_buf.data(),
                                                          imu_buf.size()),
                        opts_.gravity_window_samples,
                        opts_.stationary_max_score);
                    if (!window.found) {
                        spdlog::warn(
                            LOG_ID
                            " Quietest IMU window still looks non-stationary "
                            "(score={:.3f} >= {:.3f}, buffer={} samples) — "
                            "gravity and gyro bias may be contaminated by "
                            "motion",
                            window.score, opts_.stationary_max_score,
                            imu_buf.size());
                    }

                    // T_body_cam.rotation(); world == first cam frame, so
                    // gravity_in_world = R_cam_body * gravity_in_body.
                    const Eigen::Matrix3d r_cam_body
                        = (**cam_ptr).T_BS.block<3, 3>(0, 0).transpose();
                    const Eigen::Vector3d g_body = MeanAccelImu(window.samples);
                    // The accel reading on a stationary IMU is the negative
                    // of gravitational acceleration (specific force points
                    // opposite to free-fall). Negate to recover the actual
                    // gravity vector.
                    state_->gravity_world = -r_cam_body * g_body;
                    state_->gravity_initialised = true;

                    // A stationary gyro's mean reading is its bias. Seed
                    // the propagated bias estimate with it so the very
                    // first preintegration (and the first-keyframe bias
                    // prior, which the factor builder centres on
                    // last_bias) starts from the measured value instead
                    // of zero. Accel bias stays zero — it is not
                    // separable from the gravity direction in a single
                    // stationary window.
                    const Eigen::Vector3d gyro_bias
                        = MeanGyroImu(window.samples);
                    state_->last_bias = gtsam::imuBias::ConstantBias(
                        Eigen::Vector3d::Zero(), gyro_bias);

                    spdlog::info(
                        LOG_ID
                        " Initialised gravity from {} IMU samples "
                        "(stationary_window_found={}, score={:.3f}, "
                        "window=[{:.3f}s, {:.3f}s]): "
                        "g_world=({:.3f},{:.3f},{:.3f}) "
                        "|g|={:.3f} m/s^2, gyro_bias=({:.5f},{:.5f},{:.5f}) "
                        "rad/s",
                        window.samples.size(), window.found, window.score,
                        static_cast<double>(window.samples.front().timestamp)
                            * 1e-9,
                        static_cast<double>(window.samples.back().timestamp)
                            * 1e-9,
                        state_->gravity_world.x(), state_->gravity_world.y(),
                        state_->gravity_world.z(), state_->gravity_world.norm(),
                        gyro_bias.x(), gyro_bias.y(), gyro_bias.z());
                } else {
                    spdlog::warn(
                        LOG_ID
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
        state_->last_keyframe_ts_ns = curr_ts;
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
    const auto& distortion_vec = (**cam_for_undistort).distortion_coefficients;
    const auto Undistort = [&](const Feature& f) {
        return UndistortLod0Pixel(f, intrinsics_vec, distortion_vec);
    };

    // Landmark association over the inlier pairs. Build the next
    // active_landmarks map fresh: a landmark observed *this* keyframe
    // stays alive only if it's actually seen — its key must be a feature
    // of the new reference frame, anything else can never be matched
    // again.
    std::flat_map<Feature, LandmarkId> next_active;

    // 3D positions for the pairs that did survive triangulation. Only
    // feeds the (diagnostic) world position of new landmarks — with
    // smart factors the graph never consumes it.
    std::flat_map<Feature, const Landmark*> tri_by_curr;
    for (const auto& lm : tri.landmarks) {
        tri_by_curr.emplace(lm.feat_curr, &lm);
    }

    delta.observations.reserve(2 * pairs.size());
    delta.new_landmarks_world.reserve(pairs.size());

    // A reference feature matched by two current features (possible
    // across adjacent LODs) would otherwise spawn duplicate observations
    // of one landmark at one pose.
    std::flat_set<Feature> consumed_prev;

    size_t dropped_low_parallax = 0;
    size_t reobserved_count = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& [feat_curr, feat_prev] = pairs[i];
        if (!consumed_prev.insert(feat_prev).second) {
            continue;
        }
        const auto it = state_->active_landmarks.find(feat_prev);
        if (it != state_->active_landmarks.end()) {
            // Re-observation: the landmark already has a smart factor;
            // append this keyframe's view to it downstream.
            const LandmarkId id = it->second;
            ++reobserved_count;
            next_active.insert_or_assign(feat_curr, id);
            delta.observations.push_back(LandmarkObservation{
                .pose = delta.pose_id,
                .landmark = id,
                .pixel_lod0 = Undistort(feat_curr),
            });
            continue;
        }

        // Per-landmark parallax gate: a new landmark whose feature
        // displacement between prev and curr keyframe is below threshold
        // is depth-ambiguous; the smart factor would only ever report it
        // degenerate. Drop it rather than grow the graph with dead
        // weight. The feature may be re-acquired on a later keyframe
        // where the baseline is bigger. (`parallaxes` is index-aligned
        // with `pairs` — computed once during gating above.)
        if (parallaxes[i] < opts_.min_landmark_parallax_px) {
            ++dropped_low_parallax;
            continue;
        }
        const LandmarkId id{state_->next_landmark_id++};
        next_active.insert_or_assign(feat_curr, id);

        // World position metadata, available only when this pair was
        // also triangulated. With smart factors, landmarks are *not*
        // graph variables — they're triangulated internally by
        // SmartProjectionPoseFactor at each iSAM linearisation — so a
        // zero here affects nothing downstream.
        const auto tri_it = tri_by_curr.find(feat_curr);
        const Eigen::Vector3d p_world
            = tri_it != tri_by_curr.end()
                  ? Eigen::Vector3d(delta.R_world_cam
                                        * tri_it->second->position_cam_curr
                                    + delta.t_world_cam)
                  : Eigen::Vector3d::Zero();
        delta.new_landmarks_world.emplace_back(id, p_world);

        // For new landmarks introduced after the first keyframe we also
        // emit a feat_prev observation at the *previous* keyframe pose.
        // Two views in the graph constrain the landmark's depth and keep
        // the iSAM2 linear system well-posed without leaning on a tight
        // position prior. On the first keyframe there is no prev pose in
        // the graph, so those landmarks start single-view and pick up
        // depth on their first re-observation.
        if (delta.prev_pose_id.has_value()) {
            delta.observations.push_back(LandmarkObservation{
                .pose = delta.prev_pose_id.value(),
                .landmark = id,
                .pixel_lod0 = Undistort(feat_prev),
            });
        }
        delta.observations.push_back(LandmarkObservation{
            .pose = delta.pose_id,
            .landmark = id,
            .pixel_lod0 = Undistort(feat_curr),
        });
    }

    spdlog::info(LOG_ID
                 " Accepted keyframe pose_id={} (first={}, prev={}), "
                 "obs={}, new_landmarks={}, reobserved={}, "
                 "dropped_low_parallax={}, median_parallax={:.2f} px, "
                 "rot={:.4f} rad",
                 delta.pose_id.v, delta.is_first_keyframe,
                 delta.prev_pose_id.has_value()
                     ? std::to_string(delta.prev_pose_id.value().v)
                     : std::string{"-"},
                 delta.observations.size(), delta.new_landmarks_world.size(),
                 reobserved_count, dropped_low_parallax, median_parallax_px,
                 rot_angle);

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
        for (const auto& [_feat, id] : state_->active_landmarks) {
            known.insert(id);
        }
        for (const auto& obs : delta.observations) {
            assert(known.contains(obs.landmark));
        }
    }
#endif

    state_->active_landmarks = std::move(next_active);
    state_->last_accepted_pose = delta.pose_id;
    state_->keyframe_timestamps_ns.emplace(delta.pose_id, delta.curr_kf_ts_ns);
    if (!state_->has_origin) {
        state_->has_origin = true;
    }

    return finish(std::move(delta));
}
