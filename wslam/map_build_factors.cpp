#include "map_build_factors.hpp"

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/SmartFactorParams.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <expected>
#include <flat_map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "common.hpp"
#include "map_common.hpp"
#include "provider_base.hpp"

using namespace wslam;
using namespace wslam::map;

#define LOG_ID "[Build Factors pass]"

std::string BuildFactorsPass::getId() const { return LOG_ID; }

std::optional<std::string> BuildFactorsPass::initialize() {
    spdlog::info(LOG_ID " initializing (no-op)");
    return {};
}

namespace {
namespace impl {
// Build preintegration parameters from noise densities and the
// startup gravity estimate
std::shared_ptr<gtsam::PreintegrationCombinedParams> createIMUCalibration(
    const data::CamSensorParams& cam, const data::IMUSensorParams& imu,
    const Eigen::Vector3d& gravity, double imu_integration_sigma) {
    const double gyro_var
        = imu.gyroscope_noise_density * imu.gyroscope_noise_density;
    const double accel_var
        = imu.accelerometer_noise_density * imu.accelerometer_noise_density;
    const double gyro_bias_var
        = imu.gyroscope_random_walk * imu.gyroscope_random_walk;
    const double accel_bias_var
        = imu.accelerometer_random_walk * imu.accelerometer_random_walk;

    auto calib = gtsam::PreintegrationCombinedParams::MakeSharedU();
    calib->n_gravity = gravity;
    calib->setGyroscopeCovariance(gtsam::Matrix3::Identity() * gyro_var);
    calib->setAccelerometerCovariance(gtsam::Matrix3::Identity() * accel_var);
    calib->setIntegrationCovariance(
        gtsam::Matrix3::Identity()
        * (imu_integration_sigma * imu_integration_sigma));
    calib->setBiasOmegaCovariance(gtsam::Matrix3::Identity() * gyro_bias_var);
    calib->setBiasAccCovariance(gtsam::Matrix3::Identity() * accel_bias_var);
    calib->setBiasAccOmegaInit(gtsam::Matrix6::Identity() * 1e-5);

    // body_P_sensor = pose of the IMU in the navigation body frame, which this
    // pipeline takes to be the camera frame (the pose variables are camera
    // poses). Both extrinsics use the sensor->body T_BS convention, so:
    //   T_cam_imu = T_body_cam^-1 * T_body_imu
    // Composing imu.T_BS keeps this correct for any provider, including one
    // whose IMU frame differs from its body frame.
    //
    // Remark: a provider whose body frame coincides with its IMU frame (as in
    // EuRoC) must set imu.T_BS to identity, which collapses this to
    // T_body_cam^-1.
    const gtsam::Pose3 T_body_cam(cam.T_BS);
    const gtsam::Pose3 T_body_imu(imu.T_BS);
    calib->setBodyPSensor(T_body_cam.inverse() * T_body_imu);
    return calib;
}

// Integrate the between-keyframe IMU window into 'combined'
// Returns the number of samples integrated.
uint32_t IntegrateIMUWindow(gtsam::PreintegratedCombinedMeasurements& combined,
                            std::span<const data::IMUReading> readings,
                            uint64_t prev_kf_ts, uint64_t curr_kf_ts) {
    uint64_t last_ts = prev_kf_ts;
    uint32_t integrated = 0;

    const auto integrate
        = [&combined, &integrated](const data::IMUReading& r, double dt_s) {
              const gtsam::Vector3 accel{r.ax(), r.ay(), r.az()};
              const gtsam::Vector3 gyro{r.wx(), r.wy(), r.wz()};
              combined.integrateMeasurement(accel, gyro, dt_s);
              integrated++;
          };

    const auto is_good_ts
        = [prev_kf_ts](auto&& r) { return r.timestamp > prev_kf_ts; };

    size_t skipped_frames = 0;
    for (const auto& r : readings | std::views::filter(is_good_ts)) {
        if (r.timestamp > last_ts
            && static_cast<double>(r.timestamp - last_ts) > 1e9) {
            skipped_frames++;
        }

        const auto dt_s = static_cast<double>(r.timestamp - last_ts) * 1e-9;
        last_ts = r.timestamp;
        integrate(r, dt_s);
    }

    if (skipped_frames > 0) {
        spdlog::warn(LOG_ID " Skipped {} frames when integrating IMU",
                     skipped_frames);
    }

    // Tail up to the keyframe timestamp. Guard against an empty window —
    // readings.back() on an empty span is UB (this guard was missing before).
    if (!readings.empty() && curr_kf_ts > last_ts) {
        const double dt = static_cast<double>(curr_kf_ts - last_ts) * 1e-9;
        integrate(readings.back(), dt);
    }

    return integrated;
}
}  // namespace impl
}  // namespace

auto BuildFactorsPass::computeSensorCalibration() const -> std::expected<
    std::optional<std::tuple<decltype(imu_calibration_),
                             decltype(shared_.cam_calibration)>>,
    std::string> {
    if (shared_.cam_calibration != nullptr && imu_calibration_ != nullptr) {
        return std::nullopt;
    }

    assert(shared_.gravity_world.norm() > 1e-6);

    const auto cam_ptr = storage_.getPtr<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    if (!cam_ptr) {
        return std::unexpected{"could not get camera sensor params"};
    }

    const auto imu_ptr = storage_.getPtr<data::IMUSensorParams>(
        ResourceIdentifier::ImuParamsName);
    if (!imu_ptr) {
        return std::unexpected{"could not get imu sensor params"};
    }

    const auto& cam = **cam_ptr;
    const auto& imu = **imu_ptr;

    // Pinhole calibration shared by every smart factor. Observations are
    // pre-undistorted by the filter pass, so Cal3_S2 carries only fx,fy,cx,cy.
    auto cam_cal = std::make_shared<gtsam::Cal3_S2>(
        cam.intrinsics(0), cam.intrinsics(1), 0.0, cam.intrinsics(2),
        cam.intrinsics(3));

    auto imu_cal = impl::createIMUCalibration(cam, imu, shared_.gravity_world,
                                              opts_.integration_sigma);

    return std::make_tuple(imu_cal, cam_cal);
}

FactorBundle BuildFactorsPass::addGaugePriors(FactorBundle&& bundle,
                                              const MapChanges& delta) const {
    using gtsam::noiseModel::Diagonal;
    using gtsam::noiseModel::Isotropic;

    const auto pose_key = GtsamIdentifiers::Pose(delta.pose_id);
    const auto vel_key = GtsamIdentifiers::Velocity(delta.pose_id);
    const auto bias_key = GtsamIdentifiers::Bias(delta.pose_id);

    const gtsam::Pose3 pose_init(gtsam::Rot3(delta.R_world_cam),
                                 gtsam::Point3(delta.t_world_cam));
    bundle.new_values.insert(pose_key, pose_init);
    bundle.new_values.insert(vel_key, gtsam::Vector3(shared_.last_velocity));

    gtsam::Vector6 pose_sigmas{opts_.prior_pose_rotation_sigma_rad,
                               opts_.prior_pose_rotation_sigma_rad,
                               opts_.prior_pose_rotation_sigma_rad,
                               opts_.prior_pose_position_sigma,
                               opts_.prior_pose_position_sigma,
                               opts_.prior_pose_position_sigma};
    bundle.new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        pose_key, pose_init, Diagonal::Sigmas(pose_sigmas));

    // Loose velocity prior
    bundle.new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
        vel_key, gtsam::Vector3::Zero(),
        Isotropic::Sigma(3, opts_.prior_velocity_sigma));

    // Bias prior centred on the measured startup bias (gyro measured; accel
    // part zero, folded into the gravity estimate)
    gtsam::Vector6 bias_sigmas{
        opts_.prior_accel_bias_sigma, opts_.prior_accel_bias_sigma,
        opts_.prior_accel_bias_sigma, opts_.prior_gyro_bias_sigma,
        opts_.prior_gyro_bias_sigma,  opts_.prior_gyro_bias_sigma};

    bundle.new_factors
        .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            bias_key, shared_.last_bias, Diagonal::Sigmas(bias_sigmas));

    spdlog::debug(LOG_ID " gauge priors on x{}/v{}/b{} (first keyframe)",
                  delta.pose_id.v, delta.pose_id.v, delta.pose_id.v);

    return std::move(bundle);
}

std::expected<FactorBundle, std::string> BuildFactorsPass::integrateImu(
    FactorBundle&& bundle, const MapChanges& delta) const {
    WSLAM_ASSERT(!delta.isFirstKeyframe(), "cannot run on first frame");

    const auto pose_key = GtsamIdentifiers::Pose(delta.pose_id);
    const auto vel_key = GtsamIdentifiers::Velocity(delta.pose_id);
    const auto bias_key = GtsamIdentifiers::Bias(delta.pose_id);
    const auto prev_pose_key = GtsamIdentifiers::Pose(delta.prev_pose_id);
    const auto prev_vel_key = GtsamIdentifiers::Velocity(delta.prev_pose_id);
    const auto prev_bias_key = GtsamIdentifiers::Bias(delta.prev_pose_id);

    gtsam::PreintegratedCombinedMeasurements combined{imu_calibration_,
                                                      shared_.last_bias};
    const uint32_t integrated = impl::IntegrateIMUWindow(
        combined, std::span(delta.imu), delta.prev_kf_ts, delta.curr_kf_ts);

    bundle.new_factors.emplace_shared<gtsam::CombinedImuFactor>(
        prev_pose_key, prev_vel_key, pose_key, vel_key, prev_bias_key, bias_key,
        combined);

    // IMU-predicted NavState for the new keyframe
    const gtsam::NavState prev_nav{
        shared_.predicted_values.at<gtsam::Pose3>(prev_pose_key),
        shared_.last_velocity};
    const gtsam::NavState curr_nav
        = combined.predict(prev_nav, shared_.last_bias);

    // Hybrid initial guess: take the rotation and the metric step
    // *length* from the IMU, but keep the essential-matrix translation
    // *direction* (more reliable than short-window IMU double-integration),
    // rescaled to that step. Both factor and guess thus live in one metric
    // scale from the first edge.
    const double metric_step
        = (curr_nav.position() - prev_nav.position()).norm();
    Eigen::Vector3d t_rel = delta.t_rel;
    t_rel = t_rel.norm() > 1e-12
                ? Eigen::Vector3d(t_rel * (metric_step / t_rel.norm()))
                : Eigen::Vector3d::Zero();

    // Chain the world pose off the previous predicted pose:
    //   R_wc = R_wp * R_rel^T,  t_wc = t_wp - R_wc * t_rel
    // then take the IMU's (gyro-attitude) rotation for the actual guess.
    const Eigen::Matrix3d r_wp = prev_nav.pose().rotation().matrix();
    const Eigen::Matrix3d r_wc = r_wp * delta.R_rel.transpose();
    const Eigen::Vector3d t_wc = prev_nav.position() - r_wc * t_rel;

    // pred.v() returns a const ref into the (longer-lived) NavState; bind a ref
    // to avoid copying the velocity twice (once here, once into Values).
    const gtsam::Vector3& vel_init = curr_nav.v();
    const gtsam::Pose3 pose_init(curr_nav.pose().rotation(),
                                 gtsam::Point3(t_wc));

    bundle.new_values.insert(pose_key, pose_init);
    bundle.new_values.insert(vel_key, vel_init);

    shared_.predicted_values.update(pose_key, pose_init);

    spdlog::debug(LOG_ID
                  " CombinedImuFactor x{}-x{}: {} samples, dt={:.4f}s, "
                  "|v|={:.3f} m/s, metric_step={:.4f} m",
                  delta.prev_pose_id.v, delta.pose_id.v, integrated,
                  combined.deltaTij(), curr_nav.v().norm(), metric_step);

    return std::move(bundle);
}

FactorBundle BuildFactorsPass::addSmartFactors(FactorBundle&& bundle,
                                               const MapChanges& delta) const {
    auto noise_model
        = gtsam::noiseModel::Isotropic::Sigma(2, opts_.projection_pixel_sigma);

    gtsam::SmartProjectionParams smart_params(gtsam::HESSIAN,
                                              gtsam::ZERO_ON_DEGENERACY);
    smart_params.setRankTolerance(1e-9);
    smart_params.setLandmarkDistanceThreshold(opts_.max_landmark_distance);
    smart_params.setDynamicOutlierRejectionThreshold(
        opts_.max_observation_reprojection_px);

    // Group observations by landmark so each landmark gets exactly one factor
    // and (at most) one remove-and-readd, no matter how many poses observed it.
    std::flat_map<LandmarkId, std::vector<const Observation*>>
        landmark_observations;
    for (const auto& obs : delta.observations) {
        landmark_observations[obs.v.landmark].push_back(&obs.v);
    }

    size_t fresh = 0;
    size_t cloned = 0;

    for (const auto& [id, observations] : landmark_observations) {
        auto it = shared_.smart_factors.find(id);
        const bool is_new = it == shared_.smart_factors.end();

        SmartFactor::shared_ptr factor;
        if (is_new) {
            factor = std::make_shared<SmartFactor>(
                noise_model, shared_.cam_calibration, smart_params);
            shared_.smart_factors.insert({id, factor});
            ++fresh;
        } else {
            // Copy to avoid mutating iSAM2 state
            factor = std::make_shared<SmartFactor>(*it->second);

            const auto factor_idx_it = shared_.smart_factor_indices.find(id);
            WSLAM_ASSERT(factor_idx_it != shared_.smart_factor_indices.end(),
                         "Smart factor index exists if the factor itself "
                         "exists as well");

            bundle.remove_indices.push_back(factor_idx_it->second);
            ++cloned;
        }

        for (const auto* obs : observations) {
            factor->add(gtsam::Point2(obs->pixel.x(), obs->pixel.y()),
                        GtsamIdentifiers::Pose(obs->pose));
        }

        bundle.smart_factor_positions.emplace_back(bundle.new_factors.size(),
                                                   id);
        bundle.new_factors.push_back(factor);
    }

    spdlog::debug(
        LOG_ID " Processed smart factors: {} new, {} re-observed, {} removed",
        fresh, cloned, bundle.remove_indices.size());

    return bundle;
}

std::optional<std::string> BuildFactorsPass::execute() {
    spdlog::info(LOG_ID " Executing");

    auto calibration = computeSensorCalibration();
    if (!calibration) {
        return "calibration: " + std::move(calibration).error();
    }

    if (calibration.value().has_value()) {
        auto& [imu, cam] = calibration.value().value();
        imu_calibration_ = std::move(imu);
        shared_.cam_calibration = std::move(cam);
    }

    const auto delta_ptr
        = storage_.getPtr<MapChanges>(ResourceIdentifier::MapDeltaName);
    if (!delta_ptr) {
        return "could not get map delta";
    }
    const auto& delta = **delta_ptr;

    FactorBundle bundle;

    bundle.new_values.insert(GtsamIdentifiers::Bias(delta.pose_id),
                             shared_.last_bias);

    if (!delta.isFirstKeyframe()) {
        auto integrated = integrateImu(std::move(bundle), delta);
        if (!integrated) {
            return "imu: " + std::move(integrated).error();
        }
        bundle = std::move(integrated).value();
    } else {
        // First keyframe: no IMU edge to integrate yet, but the origin pose /
        // velocity values and the gauge priors (x/v/b) that anchor the graph
        // must still be added — otherwise the smart factors below reference a
        // pose key iSAM2 has no value for ("x1 does not exist in Values").
        bundle = addGaugePriors(std::move(bundle), delta);
    }

    bundle = addSmartFactors(std::move(bundle), delta);

    spdlog::info(LOG_ID " Built delta: {} factors, {} values, {} obs",
                 bundle.new_factors.size(), bundle.new_values.size(),
                 delta.observations.size());

    storage_.set(ResourceIdentifier::FactorBundleName, std::move(bundle));
    return {};
}
