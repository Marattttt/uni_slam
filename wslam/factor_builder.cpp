#include "factor_builder.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/SmartFactorParams.h>
#include <spdlog/spdlog.h>

#include <utility>

#include "common.hpp"
#include "models.hpp"
#include "provider_base.hpp"

using namespace wslam;

#define LOG_ID "[Factor Builder pass]"

FactorBuilderPass::FactorBuilderPass(MappingState& state,
                                     std::shared_ptr<compute::GPU> gpu,
                                     Opts opts)
    : compute::Pass(std::move(gpu)), state_(state), opts_(opts) {}

FactorBuilderPass::FactorBuilderPass(MappingState& state,
                                     std::shared_ptr<compute::GPU> gpu)
    : FactorBuilderPass(state, std::move(gpu), Opts{}) {}

std::string FactorBuilderPass::getId() const { return LOG_ID; }

std::optional<std::string> FactorBuilderPass::initialize() {
    spdlog::info(LOG_ID
                 " Initializing (pose_prior_sigma_pos={:.3g}, "
                 "between_rot_sigma={:.3g}, between_t_sigma={:.3g}, "
                 "pixel_sigma={:.3g}, robust={})",
                 opts_.prior_pose_position_sigma,
                 opts_.between_rotation_sigma_rad, opts_.between_translation_sigma,
                 opts_.projection_pixel_sigma, opts_.use_robust_projection);
    if (storage_ == nullptr) {
        return "factor builder: storage not set before initialize()";
    }
    return std::nullopt;
}

namespace {

// Integrate every sample in `imu` plus a tail from the last sample to
// `curr_kf_ts_ns` (held constant) into `pim`. Samples with non-positive
// dt are skipped — preintegration requires strictly positive intervals.
// Returns the number of samples successfully integrated (for logging).
size_t IntegrateImuWindow(gtsam::PreintegratedCombinedMeasurements& pim,
                          const std::vector<data::IMUReading>& imu,
                          uint64_t prev_kf_ts_ns, uint64_t curr_kf_ts_ns) {
    uint64_t t_last_ns = prev_kf_ts_ns;
    size_t integrated = 0;
    for (const auto& r : imu) {
        const double dt_s
            = static_cast<double>(r.timestamp - t_last_ns) * 1e-9;
        if (dt_s <= 0.0) {
            continue;
        }
        const gtsam::Vector3 accel(static_cast<double>(r.ax()),
                                   static_cast<double>(r.ay()),
                                   static_cast<double>(r.az()));
        const gtsam::Vector3 gyro(static_cast<double>(r.wx()),
                                  static_cast<double>(r.wy()),
                                  static_cast<double>(r.wz()));
        pim.integrateMeasurement(accel, gyro, dt_s);
        t_last_ns = r.timestamp;
        ++integrated;
    }
    if (!imu.empty() && curr_kf_ts_ns > t_last_ns) {
        const double dt_tail
            = static_cast<double>(curr_kf_ts_ns - t_last_ns) * 1e-9;
        const auto& last = imu.back();
        const gtsam::Vector3 accel(static_cast<double>(last.ax()),
                                   static_cast<double>(last.ay()),
                                   static_cast<double>(last.az()));
        const gtsam::Vector3 gyro(static_cast<double>(last.wx()),
                                  static_cast<double>(last.wy()),
                                  static_cast<double>(last.wz()));
        pim.integrateMeasurement(accel, gyro, dt_tail);
    }
    return integrated;
}

}  // namespace

std::optional<std::string> FactorBuilderPass::ensureImuParams() {
    if (imu_params_) {
        return std::nullopt;
    }
    if (storage_ == nullptr) {
        return "factor builder: storage pointer is null";
    }
    if (!state_.gravity_initialised) {
        return "factor builder: gravity has not been initialised yet — "
               "ensureImuParams() must run after the first accepted keyframe";
    }
    const auto imu_ptr = storage_->getPtr<data::IMUSensorParams>(
        ResourceIdentifier::ImuParamsName);
    if (!imu_ptr.has_value()) {
        return std::format("factor builder: missing IMU params under '{}'",
                           ResourceIdentifier::ImuParamsName);
    }
    const auto cam_ptr = storage_->getPtr<data::CamSensorParams>(
        ResourceIdentifier::GetCameraIntrinsicsName(0));
    if (!cam_ptr.has_value()) {
        return std::format("factor builder: missing camera params under '{}'",
                           ResourceIdentifier::GetCameraIntrinsicsName(0));
    }
    const auto& imu = **imu_ptr;
    const auto& cam = **cam_ptr;

    // Continuous-time noise → covariance for preintegration: GTSAM expects
    // the variance over a unit-time integration interval, i.e. (noise
    // density)^2. The bias *random walk* densities feed bias evolution
    // covariances directly.
    const double gyro_var = imu.gyroscope_noise_density
                            * imu.gyroscope_noise_density;
    const double accel_var = imu.accelerometer_noise_density
                             * imu.accelerometer_noise_density;
    const double gyro_bias_var = imu.gyroscope_random_walk
                                 * imu.gyroscope_random_walk;
    const double accel_bias_var = imu.accelerometer_random_walk
                                  * imu.accelerometer_random_walk;

    imu_params_ = gtsam::PreintegrationCombinedParams::MakeSharedU();
    imu_params_->n_gravity = state_.gravity_world;
    imu_params_->setGyroscopeCovariance(
        gtsam::Matrix3::Identity() * gyro_var);
    imu_params_->setAccelerometerCovariance(
        gtsam::Matrix3::Identity() * accel_var);
    imu_params_->setIntegrationCovariance(
        gtsam::Matrix3::Identity() * (opts_.integration_sigma
                                      * opts_.integration_sigma));
    imu_params_->setBiasOmegaCovariance(
        gtsam::Matrix3::Identity() * gyro_bias_var);
    imu_params_->setBiasAccCovariance(
        gtsam::Matrix3::Identity() * accel_bias_var);
    imu_params_->setBiasAccOmegaInit(gtsam::Matrix6::Identity() * 1e-5);

    // body_P_sensor tells preintegration where the IMU sits in the "body"
    // (=camera, here) frame: T_cam_imu = T_body_cam^-1 since the EuRoC
    // body frame coincides with the IMU sensor frame (IMU's T_BS is
    // identity).
    const gtsam::Pose3 T_body_cam(cam.T_BS);
    imu_params_->setBodyPSensor(T_body_cam.inverse());

    spdlog::info(LOG_ID
                 " IMU preintegration configured: |g|={:.3f} m/s^2, "
                 "T_cam_imu=({:.3f},{:.3f},{:.3f})",
                 state_.gravity_world.norm(),
                 T_body_cam.inverse().translation().x(),
                 T_body_cam.inverse().translation().y(),
                 T_body_cam.inverse().translation().z());
    return std::nullopt;
}

std::optional<std::string> FactorBuilderPass::ensureCalibration() {
    if (state_.calibration) {
        return std::nullopt;
    }
    constexpr uint32_t kCam0 = 0;
    const auto key = ResourceIdentifier::GetCameraIntrinsicsName(kCam0);
    if (storage_ == nullptr) {
        return "factor builder: storage pointer is null";
    }
    const auto cam_ptr = storage_->getPtr<data::CamSensorParams>(key);
    if (!cam_ptr.has_value()) {
        return std::format("factor builder: missing camera params under '{}'",
                           key);
    }
    state_.intrinsics_cache = *cam_ptr.value();

    // Cal3_S2(fx, fy, s, u0, v0). Observations are pre-undistorted by
    // the keyframe gate, so the smart factor's calibration only carries
    // the pinhole part. Distortion coefficients remain on
    // intrinsics_cache for the gate's undistortion step.
    const auto& cam = state_.intrinsics_cache.value();
    state_.calibration = boost::make_shared<gtsam::Cal3_S2>(
        cam.intrinsics(0), cam.intrinsics(1), 0.0, cam.intrinsics(2),
        cam.intrinsics(3));

    spdlog::info(LOG_ID
                 " Cached calibration fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f} "
                 "(distortion handled upstream: k=[{:.4f},{:.4f}] "
                 "p=[{:.4f},{:.4f}])",
                 cam.intrinsics(0), cam.intrinsics(1), cam.intrinsics(2),
                 cam.intrinsics(3), cam.distortion_coefficients(0),
                 cam.distortion_coefficients(1),
                 cam.distortion_coefficients(2),
                 cam.distortion_coefficients(3));
    return std::nullopt;
}

std::optional<std::string> FactorBuilderPass::execute() {
    spdlog::info(LOG_ID " Executing");

    new_factors_.resize(0);
    new_values_.clear();
    remove_indices_.clear();
    smart_factor_positions_.clear();
    has_work_ = false;

    if (auto err = ensureCalibration()) {
        return err;
    }

    assert(storage_ != nullptr);
    const auto delta_ptr
        = storage_->getPtr<MapDelta>(ResourceIdentifier::MapDeltaName);
    if (!delta_ptr.has_value()) {
        spdlog::warn(LOG_ID " No MapDelta in storage; nothing to build");
        return std::nullopt;
    }
    const auto& delta = *delta_ptr.value();
    if (!delta.accepted) {
        spdlog::debug(LOG_ID " MapDelta rejected upstream; nothing to build");
        return std::nullopt;
    }

    using gtsam::noiseModel::Diagonal;
    using gtsam::noiseModel::Isotropic;
    using gtsam::noiseModel::Robust;

    // Pose / velocity / bias variables for the new keyframe. The pose
    // *value* may be overwritten below with the IMU-predicted NavState
    // (see the IMU block) — we initialise here with the visual chain so
    // there's something to read if no prev frame exists.
    const auto pose_key = MappingState::poseKey(delta.pose_id);
    const auto vel_key = MappingState::velocityKey(delta.pose_id);
    const auto bias_key = MappingState::biasKey(delta.pose_id);
    gtsam::Pose3 pose_init(gtsam::Rot3(delta.R_world_cam),
                           gtsam::Point3(delta.t_world_cam));
    gtsam::Vector3 vel_init = state_.last_velocity;

    // Between-pose IMU factor — IMU gives metric translation, gravity-
    // aligned rotation, and bias evolution all in one factor. Computed
    // *before* inserting variables so the landmark-rescaling block below
    // can use the IMU-predicted pose and per-pair depth_scale.
    if (delta.prev_pose_id.has_value()) {
        if (auto err = ensureImuParams()) {
            return err;
        }
        const auto prev_pose_key
            = MappingState::poseKey(delta.prev_pose_id.value());
        const auto prev_vel_key
            = MappingState::velocityKey(delta.prev_pose_id.value());
        const auto prev_bias_key
            = MappingState::biasKey(delta.prev_pose_id.value());

        gtsam::PreintegratedCombinedMeasurements pim(imu_params_,
                                                     state_.last_bias);
        const size_t integrated = IntegrateImuWindow(pim, delta.imu_between,
                                                     delta.prev_kf_ts_ns,
                                                     delta.curr_kf_ts_ns);

        new_factors_.emplace_shared<gtsam::CombinedImuFactor>(
            prev_pose_key, prev_vel_key, pose_key, vel_key, prev_bias_key,
            bias_key, pim);

        // Vision-only odometry BetweenFactor. The IMU factor connects
        // {x_prev, v_prev, b_prev, x_curr, v_curr, b_curr} and lives in
        // a clique containing all six. iSAM2's relinearisation of a
        // *landmark* clique only includes {l, x_prev, x_curr}, so the
        // IMU information is invisible there and the depth-ambiguous
        // landmark Hessian goes singular. A loose pose-to-pose
        // BetweenFactor sits in that landmark clique too, providing the
        // local regularisation needed to keep partial Cholesky stable.
        // Visual translation is unit-norm (essential-matrix scale); the
        // metric scale comes from the IMU factor across many updates.
        const gtsam::Rot3 r_pc(delta.R_rel.transpose());
        const gtsam::Point3 t_pc(-delta.R_rel.transpose() * delta.t_rel);
        gtsam::Vector6 between_sigmas;
        between_sigmas << opts_.between_rotation_sigma_rad,
            opts_.between_rotation_sigma_rad,
            opts_.between_rotation_sigma_rad,
            opts_.between_translation_sigma,
            opts_.between_translation_sigma,
            opts_.between_translation_sigma;
        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            prev_pose_key, pose_key, gtsam::Pose3(r_pc, t_pc),
            Diagonal::Sigmas(between_sigmas));

        // pim.predict() gives a metric-scale guess for x_curr / v_curr.
        // We use only the *rotation* part as our initial guess (IMU
        // attitude is well-observable from gyro integration), and keep
        // the keyframe-gate's visual translation. Reasons:
        //   1. The visual chain is internally consistent in its own
        //      unit-norm scale — every existing landmark sits in that
        //      same scale, so re-observations will reproject correctly.
        //   2. The IMU translation is metric but during a stationary
        //      startup window |Δt_imu| ≈ 0, which would collapse the
        //      visual chain to zero motion and destroy all landmarks.
        // The CombinedImuFactor (already added above) does the actual
        // pulling toward metric scale across many iSAM updates.
        const gtsam::NavState prev_nav(
            state_.predicted_values.at<gtsam::Pose3>(prev_pose_key),
            state_.last_velocity);
        const gtsam::NavState pred = pim.predict(prev_nav, state_.last_bias);
        pose_init = gtsam::Pose3(pred.pose().rotation(), pose_init.translation());
        vel_init = pred.v();

        // Mirror the IMU-rotated pose into predicted_values too so the
        // next keyframe-gate chains off the gyro-corrected rotation.
        state_.predicted_values.update(pose_key, pose_init);

        spdlog::debug(LOG_ID
                      " Added CombinedImuFactor x{}-x{}: integrated {} "
                      "samples over dt={:.4f}s, |v_pred|={:.3f} m/s",
                      delta.prev_pose_id.value().v, delta.pose_id.v,
                      integrated, pim.deltaTij(), pred.v().norm());
    }

    new_values_.insert(pose_key, pose_init);
    new_values_.insert(vel_key, vel_init);
    new_values_.insert(bias_key, state_.last_bias);

    // First keyframe: prior on pose. Sets the gauge.
    if (delta.is_first_keyframe) {
        gtsam::Vector6 pose_sigmas;
        pose_sigmas << opts_.prior_pose_rotation_sigma_rad,
            opts_.prior_pose_rotation_sigma_rad,
            opts_.prior_pose_rotation_sigma_rad,
            opts_.prior_pose_position_sigma, opts_.prior_pose_position_sigma,
            opts_.prior_pose_position_sigma;
        new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            pose_key, pose_init, Diagonal::Sigmas(pose_sigmas));

        // Velocity prior — assume approximately stationary start.
        new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            vel_key, gtsam::Vector3::Zero(),
            Isotropic::Sigma(3, opts_.prior_velocity_sigma));
        // Bias prior — three-axis accel + three-axis gyro biases each
        // gauged with their own sigma.
        gtsam::Vector6 bias_sigmas;
        bias_sigmas << opts_.prior_accel_bias_sigma,
            opts_.prior_accel_bias_sigma, opts_.prior_accel_bias_sigma,
            opts_.prior_gyro_bias_sigma, opts_.prior_gyro_bias_sigma,
            opts_.prior_gyro_bias_sigma;
        new_factors_.emplace_shared<
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            bias_key, gtsam::imuBias::ConstantBias(),
            Diagonal::Sigmas(bias_sigmas));

        spdlog::debug(LOG_ID
                      " Added gauge priors on x{}, v{}, b{} (first keyframe)",
                      delta.pose_id.v, delta.pose_id.v, delta.pose_id.v);
    }

    // Smart projection factors — one per landmark, regardless of how
    // many keyframes observe it. The factor stores all observations and
    // internally triangulates the 3D point at every linearisation, so the
    // landmark is not a graph variable. This is the textbook VI-SLAM
    // remedy for iSAM2's partial-Cholesky failures: with no landmark
    // variable, there's no depth-axis Hessian to go singular.
    //
    // Lifecycle: a landmark observed for the first time gets a fresh
    // SmartFactor (with both prev+curr observations from the keyframe
    // gate). A re-observation of an existing landmark mutates its stored
    // SmartFactor by appending the new measurement and is then re-added
    // to iSAM2 via the remove-and-readd pattern (old factor index goes
    // into remove_indices_; mutated factor goes into new_factors_).
    // SmartProjectionPoseFactor requires a pure isotropic noise model
    // (no Robust wrapper). Outliers are handled by the dynamic-rejection
    // threshold on the smart_params instead.
    auto smart_noise = Isotropic::Sigma(2, opts_.projection_pixel_sigma);

    gtsam::SmartProjectionParams smart_params(gtsam::HESSIAN,
                                              gtsam::ZERO_ON_DEGENERACY);
    // Smallest SVD singular value below which the 2-view triangulation
    // is considered degenerate. 1e-9 is the GTSAM example default; goes
    // hand-in-hand with ZERO_ON_DEGENERACY above.
    smart_params.setRankTolerance(1e-9);
    // Maximum landmark distance (units of the world coordinate system).
    // Anything triangulated farther than this is treated as a degenerate
    // far-point — points at infinity are projectively well-defined but
    // numerically explosive in the smart factor's Jacobians and crash
    // Cal3DS2::calibrate during inverse-distortion iteration.
    smart_params.setLandmarkDistanceThreshold(50.0);
    // Reject observations whose reprojection error exceeds this many
    // sigmas after triangulation.
    smart_params.setDynamicOutlierRejectionThreshold(
        opts_.huber_threshold_px);

    // Group observations by landmark id so we can apply one
    // remove-and-readd per landmark, not one per observation.
    std::flat_map<LandmarkId, std::vector<const LandmarkObservation*>>
        obs_by_landmark;
    for (const auto& obs : delta.observations) {
        obs_by_landmark[obs.landmark].push_back(&obs);
    }

    size_t new_factor_count = 0;
    size_t reobserved_count = 0;
    for (const auto& [id, obs_list] : obs_by_landmark) {
        auto it = state_.smart_factors.find(id);
        bool is_new_factor = (it == state_.smart_factors.end());

        SmartFactor::shared_ptr factor;
        if (is_new_factor) {
            factor = boost::make_shared<SmartFactor>(smart_noise,
                                                     state_.calibration,
                                                     smart_params);
            state_.smart_factors.insert({id, factor});
            ++new_factor_count;
        } else {
            // Clone, don't mutate in place: iSAM2 still holds a
            // shared_ptr to the existing factor (it's about to be
            // removed by index below), and mutating that shared object
            // would invalidate iSAM2's bookkeeping. Add observations to
            // the clone and replace MappingState's pointer with it; the
            // old object lives on in iSAM2's graph until the worker
            // processes the remove.
            factor = boost::make_shared<SmartFactor>(*it->second);
            it->second = factor;
            auto idx_it = state_.smart_factor_indices.find(id);
            if (idx_it != state_.smart_factor_indices.end()) {
                remove_indices_.push_back(idx_it->second);
            }
            ++reobserved_count;
        }

        for (const auto* obs : obs_list) {
            factor->add(gtsam::Point2(obs->pixel_lod0.x(),
                                      obs->pixel_lod0.y()),
                        MappingState::poseKey(obs->pose));
        }

        // Record the position so the iSAM update pass can back-fill the
        // new FactorIndex into smart_factor_indices once iSAM returns.
        smart_factor_positions_.emplace_back(new_factors_.size(), id);
        new_factors_.push_back(factor);
    }

    spdlog::debug(LOG_ID
                  " Smart factors: {} new, {} re-observed (remove={})",
                  new_factor_count, reobserved_count, remove_indices_.size());

    has_work_ = true;

#ifndef NDEBUG
    // Every observation must reference either a value we just inserted in
    // this update or one that's been submitted to iSAM previously. We
    // check `predicted_values` rather than `latest_values` because the
    // worker may not have produced an optimised estimate for the most
    // recent prior keyframe yet — but predicted_values is the source of
    // truth for "has-been-submitted." iSAM2 throws if a factor references
    // an unknown key, but failing this assert gives a much clearer signal
    // at the source.
    for (const auto& obs : delta.observations) {
        const auto pk = MappingState::poseKey(obs.pose);
        const auto lk = MappingState::landmarkKey(obs.landmark);
        const bool pose_known = new_values_.exists(pk)
                                 || state_.predicted_values.exists(pk);
        const bool lm_known = new_values_.exists(lk)
                               || state_.predicted_values.exists(lk);
        assert(pose_known && "projection factor references unknown pose key");
        assert(lm_known && "projection factor references unknown landmark key");
    }
#endif

    spdlog::info(LOG_ID
                 " Built delta: {} factors, {} new values, {} obs",
                 new_factors_.size(), new_values_.size(),
                 delta.observations.size());

    return std::nullopt;
}
