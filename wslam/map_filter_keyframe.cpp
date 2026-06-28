#include "map_filter_keyframe.hpp"

#include <gtsam/geometry/Pose3.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <concepts>
#include <flat_set>
#include <functional>
#include <iterator>
#include <ranges>

#include "anybag.hpp"
#include "common.hpp"
#include "map_common.hpp"
#include "map_helpers_lods.hpp"
#include "models.hpp"
#include "provider_base.hpp"
#include "stage.hpp"

using namespace wslam;
using namespace wslam::map;
namespace util = wslam::map::util;

#define LOG_ID "[Filter Keyframe pass]"

std::string FilterKeyframePass::getId() const { return LOG_ID; }

std::optional<std::string> FilterKeyframePass::initialize() {
    spdlog::info(LOG_ID " Initializing (no-op)");
    return {};
}

namespace {
struct StationaryWindow {
    std::span<const data::IMUReading> samples;
    double score = std::numeric_limits<double>::infinity();
};

std::expected<StationaryWindow, std::string> FindStationaryWindow(
    std::span<const data::IMUReading> readings, size_t window_size,
    double max_score) {
    if (window_size == 0) {
        return std::unexpected("window size 0");
    }
    if (readings.size() == 0) {
        return std::unexpected("buffer size 0");
    }

    // Quietness of one window: population standard deviation of the
    // acceleration magnitude, plus the mean angular-velocity deviation from
    // the window's own mean. Subtracting the window mean makes the gyroscope
    // term immune to a constant bias, and using acceleration *magnitude*
    // makes the accelerometer term immune to a constant gravity vector, so a
    // parked vehicle scores near zero.
    const auto score_window
        = [](std::span<const data::IMUReading> window) -> double {
        const auto sample_count = static_cast<double>(window.size());

        Eigen::Vector3d mean_angular_velocity = Eigen::Vector3d::Zero();
        double acceleration_magnitude_sum = 0.0;
        double acceleration_magnitude_squared_sum = 0.0;
        for (const data::IMUReading& reading : window) {
            mean_angular_velocity
                += Eigen::Vector3d(static_cast<double>(reading.wx()),
                                   static_cast<double>(reading.wy()),
                                   static_cast<double>(reading.wz()));
            const double acceleration_magnitude
                = Eigen::Vector3d(static_cast<double>(reading.ax()),
                                  static_cast<double>(reading.ay()),
                                  static_cast<double>(reading.az()))
                      .norm();
            acceleration_magnitude_sum += acceleration_magnitude;
            acceleration_magnitude_squared_sum
                += acceleration_magnitude * acceleration_magnitude;
        }
        mean_angular_velocity /= sample_count;

        double angular_velocity_deviation_sum = 0.0;
        for (const data::IMUReading& reading : window) {
            const Eigen::Vector3d angular_velocity(
                static_cast<double>(reading.wx()),
                static_cast<double>(reading.wy()),
                static_cast<double>(reading.wz()));
            angular_velocity_deviation_sum
                += (angular_velocity - mean_angular_velocity).norm();
        }

        const double mean_acceleration_magnitude
            = acceleration_magnitude_sum / sample_count;
        const double acceleration_magnitude_variance
            = std::max(0.0, acceleration_magnitude_squared_sum / sample_count
                                - mean_acceleration_magnitude
                                      * mean_acceleration_magnitude);

        return std::sqrt(acceleration_magnitude_variance)
               + angular_velocity_deviation_sum / sample_count;
    };

    const auto step = std::max(1UZ, window_size / 2);
    const auto windows
        = readings | std::views::slide(window_size) | std::views::stride(step);

    StationaryWindow best{.samples = readings};
    for (const auto& chunk : windows) {
        // chunk leaves a short final window when the buffer isn't an exact
        // multiple of window_size. A one- or two-sample window has almost no
        // spread and would win with a meaningless score, so skip it -- unless
        // the buffer is smaller than a single window, in which case that lone
        // short chunk *is* the whole buffer and we score it, recovering the
        // old "average everything" fallback.
        const bool is_full_window = chunk.size() == window_size;
        if (!is_full_window && readings.size() >= window_size) {
            continue;
        }
        const std::span<const data::IMUReading> window(chunk);
        const double score = score_window(window);
        if (score < best.score) {
            best = {.samples = window, .score = score};
        }
    }

    if (best.score >= max_score) {
        return std::unexpected("no window quiet enough to be stationary");
    }
    return best;
}
};  // namespace

std::optional<std::string> FilterKeyframePass::execute() {
    spdlog::info(LOG_ID " Executing");

    auto triangulated = processTriangulation().transform_error(
        [](auto&& err) { return "triangulation: " + err; });

    if (!triangulated) {
        spdlog::warn(LOG_ID " {}", std::move(triangulated).error());
        return compute::kStageStopExecution;
    }
    auto parallax = collectMatches().transform_error(
        [](auto&& err) { return "parallax: " + err; });

    if (!parallax) {
        spdlog::warn(LOG_ID " {}", std::move(parallax).error());
        return compute::kStageStopExecution;
    }

    if (!shouldAcceptMatches(triangulated.value(), parallax.value())) {
        return compute::kStageStopExecution;
    }

    auto delta = initializeDelta(triangulated.value());

    if (auto err = setDeltaTimestamps(delta)) {
        return "setting timestamps: " + std::move(err).value();
    }

    if (auto err = processIMU(delta)) {
        return "imu: " + std::move(err).value();
    }

    auto processed = processLandmarks(std::move(delta), parallax.value(),
                                      triangulated.value());
    if (!processed) {
        return "landmarks: " + std::move(processed).error();
    }

    auto& [processed_delta, next_active_landmarks] = processed.value();

    delta = std::move(processed_delta);

    shared_.keyframes_processed++;
    shared_.active_landmarks = std::move(next_active_landmarks);
    shared_.last_pose = delta.pose_id;
    shared_.last_kf_ts = delta.curr_kf_ts;
    shared_.pose_timestamps.insert_or_assign(delta.pose_id, delta.curr_kf_ts);
    shared_.predicted_values.insert(
        GtsamIdentifiers::Pose(delta.pose_id),
        gtsam::Pose3{gtsam::Rot3{delta.R_world_cam},
                     gtsam::Point3{delta.t_world_cam}});

    storage_.set(ResourceIdentifier::MapDeltaName, std::move(delta));

    return {};
}

std::expected<TriangulationResult, std::string>
FilterKeyframePass::processTriangulation() const {
    const auto triangualtion_ptr = storage_.getPtr<TriangulationResult>(
        ResourceIdentifier::TriangulationResultName);

    if (!triangualtion_ptr) {
        return std::unexpected("could not access from storage");
    }

    const auto& tri = *triangualtion_ptr.value();

    if (!tri.stats.pose_recovered) {
        return std::unexpected("no pose recovered");
    }

    if (tri.landmarks.size() < filter_.min_landmarks) {
        return std::unexpected("not enough landmarks");
    }

    return tri;
}

MapChanges FilterKeyframePass::initializeDelta(
    const TriangulationResult& tri) const {
    bool is_first_kf = shared_.keyframes_processed == 0;
    MapChanges delta{};
    delta.pose_id = PoseId{shared_.keyframes_processed};

    if (is_first_kf) {
        return delta;
    }

    assert(shared_.last_pose.v > 0 && "last pose id is not empty");
    delta.prev_pose_id = shared_.last_pose;
    delta.prev_kf_ts = shared_.last_kf_ts;

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
    const auto prev_key = GtsamIdentifiers::Pose(delta.prev_pose_id);

    const auto& prev_pose = shared_.predicted_values.at<gtsam::Pose3>(prev_key);
    const Eigen::Matrix3d r_wp = prev_pose.rotation().matrix();
    const Eigen::Vector3d t_wp = prev_pose.translation();
    const Eigen::Matrix3d r_wc = r_wp * tri.rotation.transpose();
    const Eigen::Vector3d t_wc = t_wp - r_wc * tri.translation;
    delta.R_world_cam = r_wc;
    delta.t_world_cam = t_wc;
    delta.R_rel = tri.rotation;
    delta.t_rel = tri.translation;

    return delta;
}

namespace {
std::expected<const MatchResult*, std::string> getRansacInliers(
    const AnyBag& storage) {
    const auto ransac
        = storage.getPtr<RansacResult>(ResourceIdentifier::RansacResultName);

    if (!ransac) {
        return std::unexpected("no ransac result in storage");
    }

    std::vector<FeaturePair> feat_pairs;

    const auto& inliers = ransac.value()->inliers;

    const auto total_pairs = std::ranges::fold_left(
        inliers, 0UZ, [](auto acc, auto&& lod) { return acc + lod.size(); });

    if (total_pairs == 0) {
        return std::unexpected("ransac reports 0 pairs");
    }

    return &inliers;
}

double getParallax(const Feature& feat, const Feature& against) {
    const double factor = util::LodScale(feat.lod);

    const Eigen::Vector2d coords_feat
        = Eigen::Vector2d(feat.x, feat.y) * factor;
    const Eigen::Vector2d coords_against
        = Eigen::Vector2d{against.x, against.y} * factor;

    const double parallax = (coords_feat - coords_against).norm();

    return parallax;
}
};  // namespace

auto FilterKeyframePass::collectMatches() const
    -> std::expected<MatchInfo, std::string> {
    const auto inliers = getRansacInliers(storage_).transform_error(
        [](auto&& err) { return "getting inliers: " + err; });

    if (!inliers) {
        return std::unexpected(std::move(inliers).error());
    }

    MatchInfo info;
    std::vector<double> parallaxes;

    const auto is_tracked = [&](const Feature& feat) {
        return shared_.active_landmarks.contains(feat);
    };

    for (const auto& lod : *inliers.value()) {
        for (const auto& [feat, prev] : lod) {
            double par = getParallax(feat, prev);

            if (par < filter_.landmark_parallax_filter_px) {
                continue;
            }

            parallaxes.emplace_back(par);

            if (is_tracked(prev)) {
                const auto landmark_id = shared_.active_landmarks.at(prev);
                info.tracked_matches.emplace_back(std::make_pair(feat, prev),
                                                  landmark_id);
            } else {
                info.new_matches.emplace_back(feat, prev);
            }
        }
    }

    // Median on a scratch copy — nth_element reorders and avoids sorting the
    // entire range
    const auto mid = parallaxes.begin()
                     + static_cast<std::ptrdiff_t>(parallaxes.size() / 2);
    std::nth_element(parallaxes.begin(), mid, parallaxes.end());

    info.median_parallax = *mid;

    return info;
}

bool FilterKeyframePass::shouldAcceptMatches(const TriangulationResult& triang,
                                             const MatchInfo& match) const {
    const double rotation = util::ComputeRotationAngle(triang.rotation);

    if (shared_.keyframes_processed == 0) {
        if (match.median_parallax < filter_.bootstrap_min_pallax_px) {
            spdlog::debug(
                LOG_ID
                " Bootstrap: parallax median {:.2f} < {:.2f}; not anchoring",
                match.median_parallax, filter_.bootstrap_min_pallax_px);
            return false;
        }

        spdlog::info(LOG_ID
                     " Attempt bootstrapping on parallax median {:.2f} and "
                     "rotation {:.2f}",
                     match.median_parallax, rotation);
    }

    const bool is_small_motion
        = (rotation < filter_.min_rotation_rad)
          && (match.median_parallax < filter_.min_parallax_px);

    if (is_small_motion) {
        spdlog::debug(LOG_ID " Motion too small");
        return false;
    }

    if (match.new_matches.size() < filter_.min_new_landmarks) {
        spdlog::debug(LOG_ID " Not enough new landmarks. {} < {}",
                      match.new_matches.size(), filter_.min_new_landmarks);
        return false;
    }

    return true;
}

std::optional<std::string> FilterKeyframePass::processIMU(
    MapChanges& delta) const {
    auto imu_buf = storage_.take<std::vector<data::IMUReading>>(
        ResourceIdentifier::GetImuVecName());

    if (!imu_buf) {
        return "could not take imu buf";
    }

    if (delta.isFirstKeyframe()) {
        if (auto err = initializeFirstFrame(std::span(imu_buf.value()))) {
            return "initializing first keyframe: " + std::move(err).value();
        }
    }

    const uint64_t from = delta.prev_kf_ts;
    const uint64_t to = delta.curr_kf_ts;

    auto first = std::ranges::find_if(
        imu_buf.value(), [&](auto& imu) { return imu.timestamp >= from; });

    auto last = std::find_if(first, imu_buf->end(),
                             [&](auto& imu) { return imu.timestamp >= to; });

    // Put back into starage values not consumed until current keyframe
    std::vector<data::IMUReading> remainder{last, imu_buf->end()};
    storage_.set(ResourceIdentifier::GetImuVecName(), std::move(remainder));

    // Move and resize with no copy if valid data starts from the start, copy
    // otherwise
    if (first == imu_buf->begin()) {
        delta.imu = std::move(imu_buf).value();

        const auto relevant = static_cast<size_t>(std::distance(first, last));
        delta.imu.resize(relevant);
    } else {
        delta.imu = std::vector<data::IMUReading>{first, last};
    }

    return {};
}

std::optional<std::string> FilterKeyframePass::setDeltaTimestamps(
    MapChanges& delta) const {
    const auto curr_kf_ts
        = storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName);
    if (!curr_kf_ts) {
        return "could not get current keyframe timestamp";
    }

    delta.curr_kf_ts = curr_kf_ts.value();
    delta.prev_kf_ts = shared_.last_kf_ts;

    return {};
}

namespace {
std::pair<Eigen::Vector3d, Eigen::Vector3d> MeanAccelGyroImu(
    std::span<const data::IMUReading> readings) {
    Eigen::Vector3f accel_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_sum = Eigen::Vector3f::Zero();

    for (const auto& r : readings) {
        accel_sum += Eigen::Vector3f{r.ax(), r.ay(), r.az()};
        gyro_sum += Eigen::Vector3f{r.wx(), r.wy(), r.wz()};
    }

    Eigen::Vector3d accel = static_cast<Eigen::Vector3d>(accel_sum)
                            / static_cast<double>(readings.size());
    Eigen::Vector3d gyro = static_cast<Eigen::Vector3d>(gyro_sum)
                           / static_cast<double>(readings.size());

    return {accel, gyro};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> recoverGravityAndGyroBias(
    const StationaryWindow& window, const data::CamSensorParams& cam) {
    const Eigen::Matrix3d r_cam_body = cam.T_BS.block<3, 3>(0, 0).transpose();

    const auto [accel_mean, gyro_mean] = MeanAccelGyroImu(window.samples);

    // Acceleration of a stationary object is the negative of the gravity force
    // impacting it, so negating it and multiplying by body rotation gives a
    // gravity estimate
    const auto gravity = r_cam_body * -accel_mean;

    // Rpoorted orientation of a stationary object is its gyro bias
    const auto gyro_bias = gyro_mean;

    return {gravity, gyro_bias};
}
};  // namespace

std::optional<std::string> FilterKeyframePass::initializeFirstFrame(
    std::span<const data::IMUReading> imu_readings) const {
    const auto cam_params = storage_.getPtr<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    if (!cam_params) {
        return "could not get cam#0 sensor params";
    }

    const auto window
        = FindStationaryWindow(imu_readings, filter_.gravity_window_samples,
                               filter_.max_score_for_stationary);

    if (!window) {
        return "finding stationary window: " + std::move(window).error();
    }

    const auto [gravity, gyro_bias]
        = recoverGravityAndGyroBias(window.value(), *cam_params.value());

    shared_.gravity_world = gravity;
    shared_.last_gyro_bias = {Eigen::Vector3d::Zero(), gyro_bias};

    const auto to_seconds = [](std::integral auto num) {
        return static_cast<double>(num) * 1e-9;
    };

    const double start = to_seconds(window->samples.front().timestamp);
    const double end = to_seconds(window->samples.back().timestamp);

    spdlog::info(LOG_ID
                 " Initialized gravity from {} IMU samples; score:{:.3f} "
                 "window:[{:.3f} - {:.3f}s] gravity={:.3f} gyro_bias={:.3f}",
                 window->samples.size(), window->score, start, end,
                 gravity.mean(), gyro_bias.mean());

    return {};
}

namespace {
namespace impl {
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

using Undistorter = std::function<Eigen::Vector2d(const Feature&)>;

// Creates a temporary callable for removing distortion on features
std::expected<Undistorter, std::string> CreateUndistorter(
    const AnyBag& storage) {
    const auto params = storage.getPtr<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    if (!params) {
        return std::unexpected("could not get camera intrinsics");
    }

    return [p = params.value()](const Feature& f) -> Eigen::Vector2d {
        const auto px = util::ToLod0Pixel(f);

        const double fu = p->intrinsics[0];
        const double fv = p->intrinsics[1];
        const double cu = p->intrinsics[2];
        const double cv = p->intrinsics[3];

        const Eigen::Vector2d xy_dist((px.x() - cu) / fu, (px.y() - cv) / fv);
        const Eigen::Vector2d xy
            = UndistortNormalised(xy_dist, p->distortion_coefficients);

        return {fu * xy.x() + cu, fv * xy.y() + cv};
    };
}
}  // namespace impl
}  // namespace

std::expected<std::pair<MapChanges, std::flat_map<Feature, map::LandmarkId>>,
              std::string>
FilterKeyframePass::processLandmarks(
    MapChanges&& delta, const MatchInfo& par,
    const TriangulationResult& triangulation) const {
    assert(delta.observations.size() == 0 && "Ifield should be populated here");
    assert(delta.new_landmark_positions.size() == 0
           && "field should be populated here");

    MapChanges result = std::move(delta);

    const auto undistort = impl::CreateUndistorter(storage_);
    if (!undistort) {
        return std::unexpected("settting up undistortion: "
                               + undistort.error());
    }

    std::flat_map<Feature, LandmarkId> next_active_landmarks;
    std::flat_map<Feature, const Landmark*> tri_results;
    for (const auto& l : triangulation.landmarks) {
        tri_results.emplace(l.feat_curr, &l);
    }

    // Reserve twice the space for new matches because one insert for new
    // observation, one for a previous one to create depth
    result.observations.reserve(par.tracked_matches.size()
                                + par.new_matches.size() * 2);

    std::flat_set<Eigen::Vector2d> consumed_coords;

    for (const auto& [feat, landmark] : par.tracked_matches) {
        const auto& [curr, prev] = feat;

        if (consumed_coords.insert(util::ToLod0Pixel(prev)).second) {
            continue;
        }

        next_active_landmarks.insert_or_assign(curr, landmark);
        result.observations.emplace_back(
            Observation{.pose = delta.pose_id,
                        .landmark = landmark,
                        .pixel = undistort.value()(curr)},
            false);
    }

    for (const auto& [curr, prev] : par.new_matches) {
        shared_.last_landmark.v++;
        const LandmarkId id{shared_.last_landmark};

        next_active_landmarks.insert_or_assign(curr, id);
        result.observations.emplace_back(
            Observation{.pose = result.pose_id,
                        .landmark = id,
                        .pixel = undistort.value()(curr)},
            true);

        const auto tri_curr = tri_results.find(curr);
        if (tri_curr != tri_results.end()) {
            const Eigen::Vector3d pos_world
                = result.R_world_cam * tri_curr->second->position_cam_curr
                  + result.t_world_cam;

            result.new_landmark_positions.emplace_back(id, pos_world);
        }

        if (!result.isFirstKeyframe()) {
            result.observations.emplace_back(
                Observation{
                    .pose = result.prev_pose_id,
                    .landmark = id,
                    .pixel = undistort.value()(prev),
                },
                false);
        }
    }

    return std::make_pair(result, next_active_landmarks);
}
