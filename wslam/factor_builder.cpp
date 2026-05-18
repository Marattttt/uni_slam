#include "factor_builder.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
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

    // Cal3DS2(fx, fy, s, u0, v0, k1, k2, p1, p2)
    const auto& cam = state_.intrinsics_cache.value();
    state_.calibration = boost::make_shared<gtsam::Cal3DS2>(
        cam.intrinsics(0), cam.intrinsics(1), 0.0, cam.intrinsics(2),
        cam.intrinsics(3), cam.distortion_coefficients(0),
        cam.distortion_coefficients(1), cam.distortion_coefficients(2),
        cam.distortion_coefficients(3));

    spdlog::info(LOG_ID
                 " Cached calibration fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f} "
                 "k=[{:.4f},{:.4f}] p=[{:.4f},{:.4f}]",
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

    // Add the new pose value.
    const auto pose_key = MappingState::poseKey(delta.pose_id);
    const gtsam::Rot3 rot(delta.R_world_cam);
    const gtsam::Point3 trans(delta.t_world_cam);
    new_values_.insert(pose_key, gtsam::Pose3(rot, trans));

    // First keyframe: prior on pose. Sets the gauge.
    if (delta.is_first_keyframe) {
        gtsam::Vector6 pose_sigmas;
        pose_sigmas << opts_.prior_pose_rotation_sigma_rad,
            opts_.prior_pose_rotation_sigma_rad,
            opts_.prior_pose_rotation_sigma_rad,
            opts_.prior_pose_position_sigma, opts_.prior_pose_position_sigma,
            opts_.prior_pose_position_sigma;
        new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            pose_key, gtsam::Pose3(rot, trans), Diagonal::Sigmas(pose_sigmas));
        spdlog::debug(LOG_ID " Added gauge prior on first pose id={}",
                      delta.pose_id.v);
    }

    // Between-pose odometry factor. Skipped on the first keyframe.
    if (delta.prev_pose_id.has_value()) {
        const auto prev_key
            = MappingState::poseKey(delta.prev_pose_id.value());
        // Triangulation reports p_curr = R*p_prev + t in camera coords. The
        // BetweenFactor measures the transform T_prev_curr such that
        // T_w_curr = T_w_prev * T_prev_curr. With p_curr = R*p_prev + t we
        // have T_prev_curr's rotation = R^T and translation = -R^T * t.
        const gtsam::Rot3 r_pc(delta.R_rel.transpose());
        const gtsam::Point3 t_pc(-delta.R_rel.transpose() * delta.t_rel);
        gtsam::Vector6 sigmas;
        sigmas << opts_.between_rotation_sigma_rad,
            opts_.between_rotation_sigma_rad,
            opts_.between_rotation_sigma_rad, opts_.between_translation_sigma,
            opts_.between_translation_sigma, opts_.between_translation_sigma;
        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            prev_key, pose_key, gtsam::Pose3(r_pc, t_pc),
            Diagonal::Sigmas(sigmas));
    }

    // Add new landmark variables.
    //
    // On the first keyframe the prev camera pose is not in the graph (it
    // pre-dates the gauge), so each new landmark has only one projection
    // observation — underdetermined in depth. We therefore anchor every
    // first-keyframe landmark with a tight prior set to its triangulated
    // position. This freezes the initial map up to the monocular scale,
    // which the scale_lock prior on the first landmark fixes.
    //
    // On subsequent keyframes the keyframe gate already arranges for every
    // newly introduced landmark to have observations from both the prev and
    // current poses (see KeyframeGatePass), so the linear system is well
    // posed without any landmark prior at all.
    bool first_landmark_priored = false;
    for (const auto& [id, p_world] : delta.new_landmarks_world) {
        const auto k = MappingState::landmarkKey(id);
        new_values_.insert(k, gtsam::Point3(p_world));

        const double sigma = delta.is_first_keyframe
                                 ? opts_.first_keyframe_landmark_sigma
                                 : opts_.landmark_anchor_sigma;
        new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
            k, gtsam::Point3(p_world), Isotropic::Sigma(3, sigma));

        if (delta.is_first_keyframe && !first_landmark_priored) {
            new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
                k, gtsam::Point3(p_world),
                Isotropic::Sigma(3, opts_.prior_landmark_sigma));
            first_landmark_priored = true;
            spdlog::debug(LOG_ID
                          " Added scale-locking prior on landmark id={} at "
                          "({:.3f},{:.3f},{:.3f})",
                          id.v, p_world.x(), p_world.y(), p_world.z());
        }
    }

    // Projection factors for each observation.
    auto px_noise = Isotropic::Sigma(2, opts_.projection_pixel_sigma);
    gtsam::SharedNoiseModel obs_noise = px_noise;
    if (opts_.use_robust_projection) {
        obs_noise = Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(
                opts_.huber_threshold_px),
            px_noise);
    }

    for (const auto& obs : delta.observations) {
        const auto pk = MappingState::poseKey(obs.pose);
        const auto lk = MappingState::landmarkKey(obs.landmark);
        new_factors_.emplace_shared<gtsam::GenericProjectionFactor<
            gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>>(
            gtsam::Point2(obs.pixel_lod0.x(), obs.pixel_lod0.y()), obs_noise,
            pk, lk, state_.calibration);
    }

    has_work_ = true;

#ifndef NDEBUG
    // Every observation must reference either a value we just inserted in
    // this update or one already present in iSAM2's linearisation point.
    // iSAM2 throws if a factor references an unknown key, but failing this
    // assert gives a much clearer signal at the source.
    for (const auto& obs : delta.observations) {
        const auto pk = MappingState::poseKey(obs.pose);
        const auto lk = MappingState::landmarkKey(obs.landmark);
        const bool pose_known = new_values_.exists(pk)
                                 || state_.latest_values.exists(pk);
        const bool lm_known = new_values_.exists(lk)
                               || state_.latest_values.exists(lk);
        assert(pose_known && "projection factor references unknown pose key");
        assert(lm_known && "projection factor references unknown landmark key");
    }
#endif

    spdlog::info(LOG_ID
                 " Built delta: {} factors, {} new values (new_landmarks={}, "
                 "obs={})",
                 new_factors_.size(), new_values_.size(),
                 delta.new_landmarks_world.size(), delta.observations.size());

    return std::nullopt;
}
