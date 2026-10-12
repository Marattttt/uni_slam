#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "models.hpp"

// Use of LightGlue's per-match confidence (the model's `mscores`) in the v2
// pose estimate. The model already drops matches scoring 0.1 or less, its
// filter_threshold
namespace wslam::v2 {

struct MatchFilterOptions {
    // Matches scoring below this are dropped before RANSAC. 0 keeps all.
    // SuperVINS's value; on V101 it lowers the refined pose's p90 / max
    // direction error, and drops ~5% of the matches
    // (benchmarks/POSE_ESTIMATE_V2_NOTE.md)
    float min_score = 0.5F;
    // When fewer than this pass min_score, the best-scoring min_matches are
    // kept instead, so low-texture pairs are not starved
    size_t min_matches = 100;
};

// Matches passing the score floor, in their original order
[[nodiscard]] KeypointMatches FilterMatchesByScore(
    const KeypointMatches& matches, const MatchFilterOptions& options);

// How a match's score weights its epipolar constraint
enum class ScoreWeighting : uint8_t {
    kNone,
    kScore,
    kScoreSquared,
};

// One weight per match, empty for ScoreWeighting::kNone
[[nodiscard]] std::vector<double> MatchWeights(const KeypointMatches& matches,
                                               ScoreWeighting weighting);

}  // namespace wslam::v2
