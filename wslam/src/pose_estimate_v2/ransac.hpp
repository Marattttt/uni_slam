#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "util/epipolar_geometry.hpp"

namespace wslam::v2 {

// Filters the model's correspondences with adaptive RANSAC over eight-point
// fundamental-matrix fits and a Sampson inlier test. Fits on undistorted
// pixels (K * ray), where the epipolar constraint holds; the number of
// iterations follows from the inlier ratio found (see util::RansacOptions).
//
// Input  (from AnyBag): KeypointMatches under KeypointMatchesName, camera
//                       parameters
// Output (to   AnyBag): KeypointRansacResult under KeypointRansacName
class RansacPass : public compute::Pass {
   public:
    struct Opts {
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

}  // namespace wslam::v2
