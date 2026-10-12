#include "match_scores.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <ranges>
#include <utility>

using namespace wslam;

v2::KeypointMatches v2::FilterMatchesByScore(
    const KeypointMatches& matches, const MatchFilterOptions& options) {
    const auto passes = [&](const KeypointCorrespondence& match) {
        return match.score >= options.min_score;
    };

    if (std::cmp_greater_equal(std::ranges::count_if(matches, passes),
                               options.min_matches)) {
        return matches | std::views::filter(passes)
               | std::ranges::to<KeypointMatches>();
    }

    if (matches.size() <= options.min_matches) {
        return matches;
    }

    // The best-scoring min_matches, kept in their original order. Ties at
    // the cut go to the earlier match
    std::vector<size_t> order(matches.size());
    std::ranges::iota(order, size_t{0});
    std::ranges::stable_sort(order, std::ranges::greater{},
                             [&](size_t i) { return matches[i].score; });
    order.resize(options.min_matches);
    std::ranges::sort(order);

    KeypointMatches kept;
    kept.reserve(order.size());
    for (const size_t index : order) {
        kept.push_back(matches[index]);
    }
    return kept;
}

std::vector<double> v2::MatchWeights(const KeypointMatches& matches,
                                     ScoreWeighting weighting) {
    const auto weight = [weighting](const KeypointCorrespondence& match) {
        const auto score = static_cast<double>(match.score);
        return weighting == ScoreWeighting::kScoreSquared ? score * score
                                                          : score;
    };

    if (weighting == ScoreWeighting::kNone) {
        return {};
    }
    return matches | std::views::transform(weight)
           | std::ranges::to<std::vector>();
}
