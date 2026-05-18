#include "match_features_cpu.hpp"

#include <algorithm>
#include <flat_map>
#include <utility>

#include "common.hpp"
#include "compute.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[Match Features pass]"

std::string MatchFeaturesCPU::getId() const { return LOG_ID; }

namespace {
namespace impl {

// Sums popcount(XOR) over a contiguous range [Lo, Lo + sizeof...(Is)) of
// the 8-word ORB descriptor. `std::get<N>` performs a compile-time fixed
// access — no runtime bounds check, no abstraction-induced indirection —
// so the whole helper collapses to a sequence of popcounts + add
// reduction with no branches.
template <std::size_t Lo, std::size_t... Is>
constexpr uint32_t HammingPartial(
    const std::array<uint32_t, Feature::kDescriptorBits / 32>& da,
    const std::array<uint32_t, Feature::kDescriptorBits / 32>& db,
    std::index_sequence<Is...> /*tag*/) {
    return (... + static_cast<uint32_t>(
                      std::popcount(std::get<Lo + Is>(da) ^ std::get<Lo + Is>(db))));
}

// Bounded Hamming distance for the matching hot loop. The partial sum
// over the first half of the descriptor is a lower bound on the full
// distance (popcount contributions are non-negative): if it already
// meets or exceeds the running cap, no further accumulation can improve
// on it. Returning the cap is enough for the caller to reject the
// candidate. This roughly halves the work for unrelated descriptor
// pairs, which dominate the candidate set.
constexpr uint32_t HammingDistanceBounded(const Feature& a, const Feature& b,
                                          uint32_t cap) {
    static constexpr std::size_t kWords = Feature::kDescriptorBits / 32;
    static constexpr std::size_t kHalf = kWords / 2;
    const uint32_t lower = HammingPartial<0>(
        a.descriptor, b.descriptor, std::make_index_sequence<kHalf>{});
    if (lower >= cap) {
        return cap;
    }
    const uint32_t upper = HammingPartial<kHalf>(
        a.descriptor, b.descriptor,
        std::make_index_sequence<kWords - kHalf>{});
    return lower + upper;
}

// Returns the LOD-0 pixel coordinates of a feature.
constexpr std::pair<float, float> toLod0(const Feature& f) {
    float scale = 1.0F;
    for (uint32_t i = 0; i < f.lod; ++i) {
        scale *= static_cast<float>(GPUConst::lod_scale_factor);
    }
    return {static_cast<float>(f.x) * scale, static_cast<float>(f.y) * scale};
}

std::vector<std::pair<Feature, Feature>> FindMatchesInLod(
    const std::vector<Feature>& set_a, const std::vector<Feature>& set_b) {
    static constexpr float kLoweTestThreshold = 0.75F;
    static constexpr float kMaxDistX = 0.1F * GPUConst::frame_width;
    static constexpr float kMaxDistY = 0.1F * GPUConst::frame_height;

    std::vector<std::pair<Feature, Feature>> matches;
    if (set_b.size() < 2) {
        return matches;
    }

    // Pre-compute LOD-0 pixel coordinates of every candidate once, instead
    // of recomputing them inside the per-query spatial-window check. The
    // pyramid scale factor only depends on `lod`, which is fixed per
    // feature, so this is pure CSE — but doing it across the whole set
    // also lets the spatial gate skip Hamming work entirely for any
    // candidate outside the search window.
    std::vector<std::pair<float, float>> b_px;
    b_px.reserve(set_b.size());
    for (const auto& f : set_b) {
        b_px.push_back(toLod0(f));
    }

    matches.reserve(set_a.size());
    for (const auto& curr : set_a) {
        const auto [cx, cy] = toLod0(curr);
        uint32_t best_dist = std::numeric_limits<uint32_t>::max();
        uint32_t second_best_dist = std::numeric_limits<uint32_t>::max();
        size_t best_idx = 0;
        bool best_found = false;
        for (size_t i = 0; i < set_b.size(); i++) {
            const auto& [bx, by] = b_px.at(i);
            // Spatial pre-filter: any feature outside the search window
            // both wastes Hamming cycles and skews the Lowe ratio test,
            // since a distant but descriptor-similar feature can become
            // an artificially good "second best". Gating here is cheap.
            if (std::abs(cx - bx) > kMaxDistX
                || std::abs(cy - by) > kMaxDistY) {
                continue;
            }
            const auto distance
                = HammingDistanceBounded(curr, set_b.at(i), second_best_dist);
            if (distance < best_dist) {
                second_best_dist = best_dist;
                best_dist = distance;
                best_idx = i;
                best_found = true;
            } else if (distance < second_best_dist) {
                second_best_dist = distance;
            }
        }
        if (best_found
            && static_cast<float>(best_dist)
                   < kLoweTestThreshold
                         * static_cast<float>(second_best_dist)) {
            matches.emplace_back(curr, set_b.at(best_idx));
        }
    }
    return matches;
}

MatchResult FindMatches(const FeatureSet& set_a, const FeatureSet& set_b) {
    constexpr size_t kNumLods = GPUConst::levels_of_detail;
    const auto adjacent = [](const FeatureSet& set, size_t lod) {
        std::vector<Feature> out;
        if (lod > 0) {
            out.reserve(set.at(lod - 1).size()
                        + (lod + 1 < kNumLods ? set.at(lod + 1).size() : 0));
            std::ranges::copy(set.at(lod - 1), std::back_inserter(out));
        }
        if (lod + 1 < kNumLods) {
            std::ranges::copy(set.at(lod + 1), std::back_inserter(out));
        }
        return out;
    };

    // Build the adjacent sets once per direction; FindMatchesInLod
    // consumes the same vectors in both the A→B and B→A passes.
    std::array<std::vector<Feature>, kNumLods> adj_a;
    std::array<std::vector<Feature>, kNumLods> adj_b;
    for (size_t lod = 0; lod < kNumLods; ++lod) {
        adj_a.at(lod) = adjacent(set_a, lod);
        adj_b.at(lod) = adjacent(set_b, lod);
    }

    std::flat_map<Feature, Feature> b_to_a;
    for (size_t lod = 0; lod < kNumLods; lod++) {
        for (auto&& [b, a] :
             FindMatchesInLod(set_b.at(lod), adj_a.at(lod))) {
            b_to_a.emplace(b, a);
        }
    }
    // result[lod]: key = current-frame feature (b), value = prev-frame feature (a)
    MatchResult result;
    for (size_t lod = 0; lod < kNumLods; lod++) {
        for (auto&& [a, b] :
             FindMatchesInLod(set_a.at(lod), adj_b.at(lod))) {
            const auto it = b_to_a.find(b);
            if (it != b_to_a.end() && it->second == a) {
                result.at(lod).emplace(b, a);
            }
        }
    }
    return result;
}
};  // namespace impl
}  // namespace

std::optional<std::string> MatchFeaturesCPU::initialize() {
    spdlog::info(LOG_ID " Initializing");

    return {};
}

std::optional<std::string> MatchFeaturesCPU::execute() {
    spdlog::info(LOG_ID " Exeecuting");

    const auto count_features = [](auto&& vec) {
        return std::ranges::fold_left(vec, 0UZ, [](size_t prev, auto&& vec) {
            return prev + vec.size();
        });
    };
    auto& storage = shared_.getStorage();

    const auto prev
        = storage.getPtr<FeatureSet>(ResourceIdentifier::GetFeatureSetName(1));

    if (!prev) {
        spdlog::warn(LOG_ID
                     " Could not get featureset for previous frame. "
                     "Stopping execution");
        return compute::kComputeStopExecution;
    }

    const auto curr
        = storage.getPtr<FeatureSet>(ResourceIdentifier::GetFeatureSetName(0));
    assert(curr);

    size_t prev_feat_count = count_features(*prev.value());
    size_t curr_feat_count = count_features(*curr.value());

    if (prev_feat_count == 0) {
        spdlog::warn(LOG_ID " 0 features in featureset for previous frame");
    }

    if (curr_feat_count == 0) {
        spdlog::warn(LOG_ID " 0 features in featureset for current frame");
    }

    auto matches = impl::FindMatches(**prev, **curr);

    const auto count_matches = [](const MatchResult& m) {
        return std::ranges::fold_left(m, 0UZ, [](size_t acc, const auto& map) {
            return acc + map.size();
        });
    };

    spdlog::info(LOG_ID
                 " Finished comparing feature sets. {} against {}; common:{}",
                 prev_feat_count, curr_feat_count, count_matches(matches));

    storage.set(ResourceIdentifier::MatchedFeaturesName, std::move(matches));

    return {};
}
