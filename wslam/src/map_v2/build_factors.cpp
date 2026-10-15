#include "build_factors.hpp"

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/SmartFactorParams.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <flat_map>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "map/utils.hpp"
#include "models.hpp"
#include "pose_estimate_v2/util/camera.hpp"

using namespace wslam;
using namespace wslam::v2::mapping;
namespace utils = wslam::map::utils;

#define LOG_ID "[Build Factors v2 pass]"

std::string BuildFactorsPass::getId() const { return LOG_ID; }

std::optional<std::string> BuildFactorsPass::initialize() {
    spdlog::info(LOG_ID " Initializing (no-op)");
    return std::nullopt;
}

std::optional<std::string> BuildFactorsPass::ensureCalibration() {
    if (shared_.cam_calibration != nullptr && imu_calibration_ != nullptr) {
        return std::nullopt;
    }

    WSLAM_ASSERT(shared_.gravity_world.norm() > 1e-6,
                 "gravity is initialised with the first keyframe");

    auto camera = v2::util::ReadCameraParams(storage_, opts_.camera_index);
    if (!camera) {
        return std::move(camera).error();
    }
    const auto imu = storage_.getPtr<data::IMUSensorParams>(
        ResourceIdentifier::ImuParamsName);
    if (!imu) {
        return "no IMU parameters";
    }

    // Observations are undistorted upstream, so the smart factors see a
    // pinhole camera
    const Eigen::Vector4d& k = camera->intrinsics;  // fu, fv, cu, cv
    shared_.cam_calibration
        = std::make_shared<gtsam::Cal3_S2>(k[0], k[1], 0.0, k[2], k[3]);
    imu_calibration_ = utils::CreateImuCalibration(
        *camera, *imu.value(), shared_.gravity_world, opts_.integration_sigma);

    return std::nullopt;
}

void BuildFactorsPass::addGaugePriors(FactorBundle& bundle,
                                      const MapChanges& changes) const {
    using gtsam::noiseModel::Diagonal;
    using gtsam::noiseModel::Isotropic;

    const auto pose_key = Pose(changes.pose_id);
    const auto velocity_key = Velocity(changes.pose_id);
    const auto bias_key = Bias(changes.pose_id);

    const gtsam::Pose3 pose_init(gtsam::Rot3(changes.R_world_cam),
                                 gtsam::Point3(changes.t_world_cam));
    bundle.new_values.insert(pose_key, pose_init);
    bundle.new_values.insert(velocity_key,
                             gtsam::Vector3(shared_.last_velocity));

    // Pose3's rotation tangent is in the body frame: the gravity axis there
    // gets the gauge sigma, the two axes across it the tilt sigma
    const double yaw_variance = opts_.prior_pose_rotation_sigma_rad
                                * opts_.prior_pose_rotation_sigma_rad;
    const double tilt_variance = std::max(
        opts_.prior_pose_tilt_sigma_rad * opts_.prior_pose_tilt_sigma_rad,
        yaw_variance);
    const Eigen::Vector3d up = pose_init.rotation().unrotate(
        gtsam::Point3(shared_.gravity_world.normalized()));
    const Eigen::Matrix3d along_gravity = up * up.transpose();
    gtsam::Matrix6 pose_covariance = gtsam::Matrix6::Zero();
    pose_covariance.topLeftCorner<3, 3>()
        = yaw_variance * along_gravity
          + tilt_variance * (Eigen::Matrix3d::Identity() - along_gravity);
    pose_covariance.bottomRightCorner<3, 3>()
        = Eigen::Matrix3d::Identity()
          * (opts_.prior_pose_position_sigma * opts_.prior_pose_position_sigma);
    bundle.new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        pose_key, pose_init,
        gtsam::noiseModel::Gaussian::Covariance(pose_covariance));

    const double velocity_sigma = shared_.initial_velocity_sigma > 0.0
                                      ? shared_.initial_velocity_sigma
                                      : opts_.prior_velocity_sigma;
    bundle.new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
        velocity_key, gtsam::Vector3(shared_.last_velocity),
        Isotropic::Sigma(3, velocity_sigma));

    // Centred on the measured startup bias (gyroscope measured; accelerometer
    // along gravity only, the rest is folded into the gravity estimate)
    const gtsam::Vector6 bias_sigmas{
        opts_.prior_accel_bias_sigma, opts_.prior_accel_bias_sigma,
        opts_.prior_accel_bias_sigma, opts_.prior_gyro_bias_sigma,
        opts_.prior_gyro_bias_sigma,  opts_.prior_gyro_bias_sigma};
    bundle.new_factors
        .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            bias_key, shared_.last_bias, Diagonal::Sigmas(bias_sigmas));

    spdlog::info(LOG_ID
                 " gauge priors on x{}/v{}/b{} (first keyframe): tilt sigma "
                 "{:.2f} deg, velocity ({:.3f}, {:.3f}, {:.3f}) sigma {:.3f} "
                 "m/s",
                 changes.pose_id.v, changes.pose_id.v, changes.pose_id.v,
                 std::sqrt(tilt_variance) * 180.0 / M_PI,
                 shared_.last_velocity.x(), shared_.last_velocity.y(),
                 shared_.last_velocity.z(), velocity_sigma);
}

void BuildFactorsPass::integrateImu(FactorBundle& bundle,
                                    const MapChanges& changes) const {
    WSLAM_ASSERT(!changes.isFirstKeyframe(), "no IMU edge on the first frame");

    const auto pose_key = Pose(changes.pose_id);
    const auto velocity_key = Velocity(changes.pose_id);
    const auto bias_key = Bias(changes.pose_id);
    const auto prev_pose_key = Pose(changes.prev_pose_id);
    const auto prev_velocity_key = Velocity(changes.prev_pose_id);
    const auto prev_bias_key = Bias(changes.prev_pose_id);

    gtsam::PreintegratedCombinedMeasurements combined{imu_calibration_,
                                                      shared_.last_bias};
    const uint32_t integrated = utils::IntegrateImuWindow(
        combined, changes.imu, changes.prev_kf_ts, changes.curr_kf_ts);

    bundle.new_factors.emplace_shared<gtsam::CombinedImuFactor>(
        prev_pose_key, prev_velocity_key, pose_key, velocity_key, prev_bias_key,
        bias_key, combined);

    // IMU-predicted state of the new keyframe
    const gtsam::NavState prev_nav{
        shared_.predicted_values.at<gtsam::Pose3>(prev_pose_key),
        shared_.last_velocity};
    const gtsam::NavState curr_nav
        = combined.predict(prev_nav, shared_.last_bias);

    // Hybrid initial guess: rotation and metric step length from the IMU,
    // translation direction from the essential matrix (more reliable than
    // short-window double integration), rescaled to that step
    const double metric_step
        = (curr_nav.position() - prev_nav.position()).norm();
    const Eigen::Vector3d t_rel
        = changes.t_rel.norm() > 1e-12
              ? Eigen::Vector3d(changes.t_rel
                                * (metric_step / changes.t_rel.norm()))
              : Eigen::Vector3d::Zero();

    const Eigen::Matrix3d r_wp = prev_nav.pose().rotation().matrix();
    const Eigen::Matrix3d r_wc = r_wp * changes.R_rel.transpose();
    const Eigen::Vector3d t_wc = prev_nav.position() - r_wc * t_rel;

    const gtsam::Pose3 pose_init(curr_nav.pose().rotation(),
                                 gtsam::Point3(t_wc));
    bundle.new_values.insert(pose_key, pose_init);
    bundle.new_values.insert(velocity_key, curr_nav.v());

    shared_.predicted_values.update(pose_key, pose_init);
    shared_.predicted_velocities.insert_or_assign(changes.pose_id,
                                                  curr_nav.v());
    publishMotionPrior(changes, curr_nav, metric_step);

    spdlog::debug(LOG_ID
                  " CombinedImuFactor x{}-x{}: {} samples, dt={:.4f}s, "
                  "|v|={:.3f} m/s, metric_step={:.4f} m",
                  changes.prev_pose_id.v, changes.pose_id.v, integrated,
                  combined.deltaTij(), curr_nav.v().norm(), metric_step);
}

void BuildFactorsPass::publishMotionPrior(const MapChanges& changes,
                                          const gtsam::NavState& state,
                                          double metric_step) const {
    // Below this step the unit-scale depths can't be made metric
    constexpr double kMinMetricStep = 1e-3;

    if (changes.median_depth_unit <= 0.0 || metric_step < kMinMetricStep) {
        spdlog::debug(LOG_ID
                      " x{}: no motion prior (depth {:.3f}, step "
                      "{:.4f} m)",
                      changes.pose_id.v, changes.median_depth_unit,
                      metric_step);
        return;
    }

    const KeyframeMotionPrior prior{
        .keyframe_ts = changes.curr_kf_ts,
        .R_world_cam = state.pose().rotation().matrix(),
        .velocity_world = state.v(),
        .gravity_world = shared_.gravity_world,
        .accel_bias = shared_.last_bias.accelerometer(),
        .gyro_bias = shared_.last_bias.gyroscope(),
        .median_depth_m = changes.median_depth_unit * metric_step,
        .velocity_sigma = shared_.velocity_error_samples > 0
                              ? std::sqrt(shared_.velocity_error_variance)
                              : 0.0,
        .min_rotation_rad = opts_.gate_min_rotation_rad,
        .min_parallax_px = opts_.gate_min_parallax_px,
    };

    spdlog::debug(LOG_ID
                  " x{}: motion prior |v| {:.3f} m/s (sigma {:.3f}), "
                  "median depth {:.2f} m",
                  changes.pose_id.v, prior.velocity_world.norm(),
                  prior.velocity_sigma, prior.median_depth_m);

    storage_.set(ResourceIdentifier::KeyframeMotionPriorName, prior);
}

void BuildFactorsPass::addSmartFactors(FactorBundle& bundle,
                                       const MapChanges& changes) const {
    const auto noise_model
        = gtsam::noiseModel::Isotropic::Sigma(2, opts_.projection_pixel_sigma);

    gtsam::SmartProjectionParams smart_params(gtsam::HESSIAN,
                                              gtsam::ZERO_ON_DEGENERACY);
    smart_params.setRankTolerance(1e-9);
    smart_params.setLandmarkDistanceThreshold(opts_.max_landmark_distance);
    smart_params.setDynamicOutlierRejectionThreshold(
        opts_.max_observation_reprojection_px);

    // One factor (and at most one remove-and-re-add) per landmark, however
    // many poses observed it this keyframe
    std::flat_map<LandmarkId, std::vector<const Observation*>>
        landmark_observations;
    for (const Observation& observation : changes.observations) {
        landmark_observations[observation.landmark].push_back(&observation);
    }

    size_t fresh = 0;
    size_t cloned = 0;

    for (const auto& [id, observations] : landmark_observations) {
        const auto it = shared_.smart_factors.find(id);

        SmartFactor::shared_ptr factor;
        if (it == shared_.smart_factors.end()) {
            factor = std::make_shared<SmartFactor>(
                noise_model, shared_.cam_calibration, smart_params);
            ++fresh;
        } else {
            // The submitted factor belongs to the smoother: add to a copy and
            // replace it
            factor = std::make_shared<SmartFactor>(*it->second);

            const auto slot = shared_.smart_factor_slots.find(id);
            WSLAM_ASSERT(slot != shared_.smart_factor_slots.end(),
                         "l{} has a factor in the window but no slot", id.v);
            bundle.remove_slots.insert(slot->second);
            ++cloned;
        }

        for (const Observation* observation : observations) {
            factor->add(
                gtsam::Point2(observation->pixel.x(), observation->pixel.y()),
                Pose(observation->pose));
        }

        shared_.smart_factors.insert_or_assign(id, factor);
        bundle.new_factors.push_back(factor);
    }

    spdlog::debug(LOG_ID " smart factors: {} new, {} re-observed", fresh,
                  cloned);
}

std::optional<std::string> BuildFactorsPass::execute() {
    spdlog::info(LOG_ID " Executing");

    if (auto err = ensureCalibration()) {
        return "calibration: " + std::move(err).value();
    }

    auto changes = storage_.take<MapChanges>(ResourceIdentifier::MapDeltaName);
    if (!changes) {
        return "no map changes";
    }

    FactorBundle bundle;
    bundle.new_values.insert(Bias(changes->pose_id), shared_.last_bias);

    const double time = SmootherTime(changes->pose_id);
    for (const gtsam::Key key :
         {Pose(changes->pose_id), Velocity(changes->pose_id),
          Bias(changes->pose_id)}) {
        bundle.timestamps.emplace(key, time);
    }

    if (changes->isFirstKeyframe()) {
        addGaugePriors(bundle, *changes);
    } else {
        integrateImu(bundle, *changes);
    }

    addSmartFactors(bundle, *changes);

    spdlog::info(LOG_ID
                 " Built delta: {} factors, {} values, {} removals, "
                 "{} observations",
                 bundle.new_factors.size(), bundle.new_values.size(),
                 bundle.remove_slots.size(), changes->observations.size());

    storage_.set(ResourceIdentifier::FactorBundleName, std::move(bundle));
    return std::nullopt;
}
