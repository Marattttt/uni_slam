#pragma once

#include <cstdint>
#include <random>

#include "common.hpp"
#include "compute/pass.hpp"

namespace wslam {

// CPU RANSAC pass that filters the previous→current feature matches produced
// by MatchFeaturesCPU using the normalised 8-point fundamental-matrix
// algorithm with a Sampson-distance inlier test.
//
// Input  (from AnyBag): MatchResult under
// ResourceIdentifier::MatchedFeaturesName Output (to   AnyBag): RansacResult
// under ResourceIdentifier::RansacResultName
class RansacCPU : public compute::Pass {
   public:
    struct Opts {
        // Maximum RANSAC iterations to run.
        uint32_t max_iterations = 500;

        // Adaptive early-exit confidence: iteration stops once the
        // probability of having already drawn at least one all-inlier
        // 8-sample (given the best model's inlier ratio) exceeds this.
        // With good matches this cuts the loop from max_iterations to a
        // few dozen draws; with poor matches it has no effect.
        double confidence = 0.99;

        // Sampson-distance threshold (in squared pixels at LOD-0) for a
        // correspondence to be considered an inlier. 3.84 ≈ chi-squared 95%
        // for 1 DoF and is a common default in the literature.
        double inlier_threshold = 3.84;

        // Below this many matches the pass writes an empty result and skips
        // the geometric fit. The 8-point algorithm needs at least 8 pairs.
        size_t min_matches = 8;

        // Seed for the internal RNG. Fixed by default so frame-to-frame
        // behaviour is reproducible.
        uint64_t rng_seed = 0xC0FFEE'1234ULL;
    };

    RansacCPU(AnyBag& storage, Opts opts)
        : storage_(storage), opts_(opts), rng_(opts.rng_seed) {}

    // Out-of-line so the Opts default member initializers (parsed only after
    // the enclosing class definition completes) can be value-initialised.
    RansacCPU(AnyBag& storage);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    AnyBag& storage_;
    Opts opts_;
    std::mt19937_64 rng_;
};
}  // namespace wslam
