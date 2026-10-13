#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "map_common.hpp"
#include "models.hpp"

namespace wslam::v2::mapping {

struct FilterKeyframeConfig {
    size_t min_landmarks = 30;
    double min_rotation_rad = 0.05;
    double min_parallax_px = 20.0;
    uint32_t min_new_landmarks = 5;
    // Inliers with less parallax than this observe no landmark
    double landmark_parallax_filter_px = 10.0;

    double bootstrap_min_parallax_px = 3;
    uint32_t gravity_window_samples = 200;
    double max_score_for_stationary = 0.8;

    // Largest distance between a keyframe keypoint's pixel as recorded and as
    // re-detected under the same index, see TrackedKeypoint
    double association_tolerance_px = 0.5;

    uint32_t camera_index = 0;
};

// Keyframe gate of the v2 mapping stage, the port of v1's FilterKeyframePass
// to the model's keypoint correspondences.
//
// Decides whether the current frame becomes a keyframe from the geometry
// between the reference (last keyframe) and the current frame. On acceptance
// it allocates the pose, chains its initial world pose off the previous
// keyframe, cuts the IMU window and, on the first keyframe, initialises
// gravity and the gyroscope bias from the quietest stationary IMU window.
// Landmark association happens later, in AssociateLandmarksPass, once the
// previous smoother update has been drained.
//
// Publishes the reference-advance marker before origin on every frame (the
// bootstrap pairs consecutive frames) and afterwards on acceptance only, so
// the model's reference stays the last keyframe.
//
// Input  (from AnyBag): KeypointTriangulationResult under
//                       KeypointTriangulationName and KeypointRansacResult
//                       under KeypointRansacName (both taken, so a frame whose
//                       pose estimate stopped early is never mapped from a
//                       previous frame's geometry), IMU buffer, frame
//                       timestamp, sensor parameters
// Output (to   AnyBag): KeyframeCandidate under KeyframeCandidateName,
//                       FeatureReferenceAdvanceName marker
class FilterKeyframePass : public compute::Pass {
   public:
    FilterKeyframePass(AnyBag& storage, SharedBindings& shared,
                       FilterKeyframeConfig config)
        : storage_(storage), shared_(shared), config_(config) {}

    [[nodiscard]] std::string getId() const override;
    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;

   private:
    struct MatchInfo {
        std::vector<CandidateMatch> matches;
        size_t tracked = 0;
        size_t pixel_mismatches = 0;
        double median_parallax = 0.0;
    };

    AnyBag& storage_;
    SharedBindings& shared_;
    const FilterKeyframeConfig config_;

    // Set by initialize()
    std::optional<data::CamSensorParams> camera_;

    [[nodiscard]] MatchInfo collectMatches(
        const KeypointRansacResult& ransac) const;

    [[nodiscard]] bool shouldAccept(
        const KeypointTriangulationResult& triangulation,
        const MatchInfo& info) const;

    [[nodiscard]] KeyframeCandidate initializeCandidate(
        const KeypointTriangulationResult& triangulation) const;

    // Moves the IMU samples up to this keyframe out of the buffer, leaving the
    // rest. On the first keyframe initialises gravity and gyroscope bias
    [[nodiscard]] std::optional<std::string> processImu(
        KeyframeCandidate& candidate) const;

    [[nodiscard]] std::optional<std::string> initializeFirstFrame(
        std::span<const data::IMUReading> imu_readings) const;
};

}  // namespace wslam::v2::mapping
