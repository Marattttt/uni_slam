#include "place_database.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/IDSelector.h>

#include <algorithm>

#include "assert.hpp"

using namespace wslam::v2::mapping::loop_closure;

struct PlaceDatabase::Index {
    explicit Index(size_t dim)
        : flat(static_cast<faiss::idx_t>(dim)), id_map(&flat) {}

    // Exact search: one vector per keyframe, so an approximate index would
    // only cost recall
    faiss::IndexFlatIP flat;
    faiss::IndexIDMap2 id_map;
};

PlaceDatabase::PlaceDatabase(size_t descriptor_dim)
    : descriptor_dim_(descriptor_dim),
      index_(std::make_unique<Index>(descriptor_dim)) {}

PlaceDatabase::~PlaceDatabase() = default;
PlaceDatabase::PlaceDatabase(PlaceDatabase&&) noexcept = default;
PlaceDatabase& PlaceDatabase::operator=(PlaceDatabase&&) noexcept = default;

void PlaceDatabase::add(uint64_t keyframe, const PlaceDescriptor& descriptor) {
    WSLAM_ASSERT(descriptor.values.size() == descriptor_dim_,
                 "adding a {}-d descriptor to a {}-d database",
                 descriptor.values.size(), descriptor_dim_);
    const auto id = static_cast<faiss::idx_t>(keyframe);
    index_->id_map.add_with_ids(1, descriptor.values.data(), &id);
}

std::vector<PlaceCandidate> PlaceDatabase::search(
    const PlaceDescriptor& query, size_t top_n, uint64_t max_keyframe) const {
    WSLAM_ASSERT(query.values.size() == descriptor_dim_,
                 "searching a {}-d database with a {}-d descriptor",
                 descriptor_dim_, query.values.size());

    const size_t k = std::min(top_n, size());
    if (k == 0) {
        return {};
    }

    // Half-open id range [0, max_keyframe + 1)
    faiss::IDSelectorRange selector(
        0, static_cast<faiss::idx_t>(max_keyframe) + 1);
    faiss::SearchParameters params;
    params.sel = &selector;

    std::vector<float> similarities(k);
    std::vector<faiss::idx_t> ids(k);
    index_->id_map.search(1, query.values.data(),
                          static_cast<faiss::idx_t>(k), similarities.data(),
                          ids.data(), &params);

    std::vector<PlaceCandidate> candidates;
    candidates.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        // Fewer than k ids passed the selector
        if (ids[i] < 0) {
            break;
        }
        candidates.push_back(PlaceCandidate{
            .keyframe = static_cast<uint64_t>(ids[i]),
            .similarity = similarities[i],
        });
    }
    return candidates;
}

size_t PlaceDatabase::size() const {
    return static_cast<size_t>(index_->id_map.ntotal);
}
