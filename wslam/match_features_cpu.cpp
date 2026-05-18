#include "match_features_cpu.hpp"

#include <algorithm>
#include <flat_map>
#include <ranges>

#include "common.hpp"
#include "compute.hpp"
#include "models.hpp"

using namespace wslam;

#define LOG_ID "[Match Features pass]"

std::string MatchFeaturesCPU::getId() const { return LOG_ID; }

namespace {
namespace impl {
template <std::integral T>
constexpr uint32_t HammingDistance(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<uint32_t>(
        std::popcount(static_cast<U>(a) ^ static_cast<U>(b)));
}
constexpr uint32_t HammingDistance(Feature a, Feature b) {
    static constinit auto indices = std::views::iota(0UZ, a.descriptor.size());

    return std::ranges::fold_left(indices, 0U, [&](auto prev, auto i) {
        return prev + HammingDistance(a.descriptor[i], b.descriptor[i]);
    });
}

// Returns the LOD-0 pixel coordinates of a feature.
constexpr std::pair<float, float> toLod0(const Feature& f) {
    float scale = 1.0F;
    for (uint32_t i = 0; i < f.lod; ++i) {
        scale *= static_cast<float>(GPUConst::lod_scale_factor);
    }
    return {static_cast<float>(f.x) * scale, static_cast<float>(f.y) * scale};
}

constexpr std::vector<std::pair<Feature, Feature>> FindMatchesInLod(
    const std::vector<Feature>& set_a, const std::vector<Feature>& set_b) {
    static constexpr float kLoweTestThreshold = 0.75F;
    static constexpr float kMaxDistX = 0.1F * GPUConst::frame_width;
    static constexpr float kMaxDistY = 0.1F * GPUConst::frame_height;

    std::vector<std::pair<Feature, Feature>> matches;
    if (set_b.size() < 2) {
        return matches;
    }
    for (const auto& curr : set_a) {
        uint32_t best_dist = std::numeric_limits<uint32_t>::max();
        uint32_t second_best_dist = std::numeric_limits<uint32_t>::max();
        size_t best_idx = 0;
        for (size_t i = 0; i < set_b.size(); i++) {
            const auto distance = HammingDistance(curr, set_b[i]);
            if (distance < best_dist) {
                second_best_dist = best_dist;
                best_dist = distance;
                best_idx = i;
            } else if (distance < second_best_dist) {
                second_best_dist = distance;
            }
        }
        if (static_cast<float>(best_dist)
            < kLoweTestThreshold * static_cast<float>(second_best_dist)) {
            const auto [cx, cy] = toLod0(curr);
            const auto [bx, by] = toLod0(set_b[best_idx]);
            if (std::abs(cx - bx) > kMaxDistX || std::abs(cy - by) > kMaxDistY) {
                continue;
            }
            matches.emplace_back(curr, set_b[best_idx]);
        }
    }
    return matches;
}

MatchResult FindMatches(const FeatureSet& set_a, const FeatureSet& set_b) {
    constexpr size_t kNumLods = GPUConst::levels_of_detail;
    auto adjacent = [](const FeatureSet& set, size_t lod) {
        std::vector<Feature> out;
        if (lod > 0) {
            std::ranges::copy(set[lod - 1], std::back_inserter(out));
        }
        if (lod + 1 < kNumLods) {
            std::ranges::copy(set[lod + 1], std::back_inserter(out));
        }
        return out;
    };
    std::flat_map<Feature, Feature> b_to_a;
    for (size_t lod = 0; lod < kNumLods; lod++) {
        for (auto&& [b, a] :
             FindMatchesInLod(set_b[lod], adjacent(set_a, lod))) {
            b_to_a.emplace(b, a);
        }
    }
    // result[lod]: key = current-frame feature (b), value = prev-frame feature (a)
    MatchResult result;
    for (size_t lod = 0; lod < kNumLods; lod++) {
        for (auto&& [a, b] :
             FindMatchesInLod(set_a[lod], adjacent(set_b, lod))) {
            const auto it = b_to_a.find(b);
            if (it != b_to_a.end() && it->second == a) {
                result[lod].emplace(b, a);
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
