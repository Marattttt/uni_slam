#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "models.hpp"
#include "pose_estimate_v2/match_scores.hpp"

using namespace wslam;

namespace {

// Matches identified by prev_index, in that order, with the given scores
v2::KeypointMatches MatchesWithScores(const std::vector<float>& scores) {
    v2::KeypointMatches matches;
    for (size_t i = 0; i < scores.size(); ++i) {
        v2::KeypointCorrespondence match{};
        match.prev_index = static_cast<uint32_t>(i);
        match.curr_index = static_cast<uint32_t>(i);
        match.score = scores[i];
        matches.push_back(match);
    }
    return matches;
}

std::vector<uint32_t> Indices(const v2::KeypointMatches& matches) {
    std::vector<uint32_t> indices;
    for (const auto& match : matches) {
        indices.push_back(match.prev_index);
    }
    return indices;
}

}  // namespace

TEST(MatchScoresTest, FloorKeepsMatchesAtOrAboveIt) {
    const auto matches = MatchesWithScores({0.9F, 0.2F, 0.5F, 0.49F, 1.0F});

    const auto kept = v2::FilterMatchesByScore(
        matches, {.min_score = 0.5F, .min_matches = 2});

    EXPECT_EQ(Indices(kept), (std::vector<uint32_t>{0, 2, 4}));
}

TEST(MatchScoresTest, ZeroFloorKeepsEverything) {
    const auto matches = MatchesWithScores({0.1F, 0.3F, 0.15F});

    const auto kept = v2::FilterMatchesByScore(matches, {.min_score = 0.0F});

    EXPECT_EQ(Indices(kept), (std::vector<uint32_t>{0, 1, 2}));
}

TEST(MatchScoresTest, FallbackKeepsTheBestScoringInOrder) {
    // Only one match passes 0.8, fewer than the 3 required
    const auto matches = MatchesWithScores({0.3F, 0.9F, 0.1F, 0.5F, 0.4F});

    const auto kept = v2::FilterMatchesByScore(
        matches, {.min_score = 0.8F, .min_matches = 3});

    EXPECT_EQ(Indices(kept), (std::vector<uint32_t>{1, 3, 4}));
}

TEST(MatchScoresTest, FallbackKeepsAllWhenThereAreTooFew) {
    const auto matches = MatchesWithScores({0.3F, 0.2F});

    const auto kept = v2::FilterMatchesByScore(
        matches, {.min_score = 0.8F, .min_matches = 5});

    EXPECT_EQ(Indices(kept), (std::vector<uint32_t>{0, 1}));
}

TEST(MatchScoresTest, WeightsFollowTheScores) {
    const auto matches = MatchesWithScores({0.5F, 1.0F});

    EXPECT_TRUE(v2::MatchWeights(matches, v2::ScoreWeighting::kNone).empty());
    EXPECT_EQ(v2::MatchWeights(matches, v2::ScoreWeighting::kScore),
              (std::vector<double>{0.5, 1.0}));
    EXPECT_EQ(v2::MatchWeights(matches, v2::ScoreWeighting::kScoreSquared),
              (std::vector<double>{0.25, 1.0}));
}
