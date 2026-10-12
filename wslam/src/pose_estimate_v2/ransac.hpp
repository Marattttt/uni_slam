#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "match_scores.hpp"
#include "models.hpp"
#include "util/epipolar_geometry.hpp"

namespace wslam::v2 {

// Filters the model's correspondences with adaptive RANSAC over eight-point
// essential-matrix fits on the undistorted rays and a Sampson inlier test in
// pixels; the number of iterations follows from the inlier ratio found (see
// util::RansacOptions). Matches below a LightGlue score floor are dropped
// first, and the scores weight the final least-squares fit (see
// match_scores.hpp).
//
// Input  (from AnyBag): KeypointMatches under KeypointMatchesName, camera
//                       parameters
// Output (to   AnyBag): KeypointRansacResult under KeypointRansacName
class RansacPass : public compute::Pass {
   public:
    struct Opts {
        MatchFilterOptions filter;
        ScoreWeighting weighting = ScoreWeighting::kScore;
        util::RansacOptions ransac;

        // Seed for the RNG, which persists across frames. Fixed so runs are
        // reproducible
        uint64_t rng_seed = 0xC0FFEE'1234ULL;

        uint32_t camera_index = 0;
    };

    RansacPass(AnyBag& storage, Opts opts)
        : storage_(storage), opts_(opts), rng_(opts.rng_seed) {}

    // Out-of-line so the Opts default member initializers (parsed only after
    // the enclosing class definition completes) can be value-initialised
    explicit RansacPass(AnyBag& storage);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    Opts opts_;
    std::mt19937_64 rng_;

    // Set by initialize()
    std::optional<data::CamSensorParams> camera_;
};

// Body of RansacPass::execute: score filter, then essential-matrix RANSAC on
// the remaining matches. Draws from `rng`
[[nodiscard]] KeypointRansacResult RunRansac(
    const KeypointMatches& matches, const data::CamSensorParams& camera,
    const RansacPass::Opts& opts, std::mt19937_64& rng);

// Mean focal length in pixels, which scales ray-space errors to pixels
[[nodiscard]] double FocalLengthPx(const data::CamSensorParams& camera);

}  // namespace wslam::v2
