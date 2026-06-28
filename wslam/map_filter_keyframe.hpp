#pragma once

#include "anybag.hpp"
#include "compute.hpp"
#include "map_common.hpp"
#include "models.hpp"
#include "pass.hpp"

namespace wslam::map {
struct FilterKeyframeConfig {
    size_t min_landmarks = 30;
    double min_rotation_rad = 0.05;
    double min_parallax_px = 20.0;
    uint32_t min_new_landmarks = 5;
    double landmark_parallax_filter_px = 10.0;

    double bootstrap_min_pallax_px = 3;
    uint32_t gravity_window_samples = 200;
    double max_score_for_stationary = 0.8;
};

class FilterKeyframePass : public compute::Pass {
   public:
    [[nodiscard]] std::string getId() const final;
    [[nodiscard]] std::optional<std::string> initialize() final;
    [[nodiscard]] std::optional<std::string> execute() final;

    FilterKeyframePass(compute::Compute& comp, MappingSharedBindings& shared,
                       FilterKeyframeConfig filter = {})
        : filter_(filter), shared_(shared), storage_(comp.getStorage()) {}

   private:
    // Information about parallax detected in a pair of (key)frames
    struct MatchInfo {
        FeaturePair pair;
        struct TrackedPair {
            FeaturePair pair;
            LandmarkId landmark;
        };

        std::vector<FeaturePair> new_matches;
        std::vector<TrackedPair> tracked_matches;
        double median_parallax;
    };

    const FilterKeyframeConfig filter_;
    MappingSharedBindings& shared_;
    AnyBag& storage_;

    // Return processable result or an error msg with a flag meaning is it a
    // critical error
    [[nodiscard]] std::expected<TriangulationResult, std::string>
    processTriangulation() const;

    // Process latest readings and report info about feature parallax in frame
    [[nodiscard]] std::expected<MatchInfo, std::string> collectMatches() const;

    // Apply filters to the match info to determine whther or not it is
    // fitting for a keyframe
    [[nodiscard]] bool shouldAcceptMatches(const TriangulationResult& triang,
                                           const MatchInfo& match) const;

    // Extract needed IMU readings from storage, leaving behind unused deta
    // On first keyframe, initializeFirstFrame is called
    [[nodiscard]] std::optional<std::string> processIMU(
        MapChanges& delta) const;

    // Set current ant previous frame's timestamps
    [[nodiscard]] std::optional<std::string> setDeltaTimestamps(
        MapChanges& delta) const;

    // Initizliize gravity estimate and first gyro bias
    [[nodiscard]] std::optional<std::string> initializeFirstFrame(
        std::span<const data::IMUReading> imu_readings) const;

    // Compare current and previous landmarks, and use data from triangulation
    // Returns updated delta and new list of active landmarks
    [[nodiscard]]
    std::expected<
        std::pair<MapChanges, std::flat_map<Feature, map::LandmarkId>>,
        std::string> processLandmarks(MapChanges&& delta, const MatchInfo& par,
                                      const TriangulationResult& triangulation)
        const;

    [[nodiscard]] MapChanges initializeDelta(
        const TriangulationResult& triangulation) const;
};
};  // namespace wslam::map
