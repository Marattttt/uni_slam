#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "model_io.hpp"

namespace wslam::v2::mapping::loop_closure {

struct PlaceCandidate {
    // Pose id of the keyframe (map_v2 PoseId value)
    uint64_t keyframe = 0;
    // Cosine similarity to the query
    float similarity = 0.0F;
};

// FAISS exact inner-product index over the keyframes' L2-normalised
// descriptors, keyed by pose id
class PlaceDatabase {
   public:
    explicit PlaceDatabase(size_t descriptor_dim);
    ~PlaceDatabase();

    PlaceDatabase(const PlaceDatabase&) = delete;
    PlaceDatabase& operator=(const PlaceDatabase&) = delete;
    PlaceDatabase(PlaceDatabase&&) noexcept;
    PlaceDatabase& operator=(PlaceDatabase&&) noexcept;

    void add(uint64_t keyframe, const PlaceDescriptor& descriptor);

    // The top_n most similar keyframes with an id of at most max_keyframe,
    // most similar first
    [[nodiscard]] std::vector<PlaceCandidate> search(
        const PlaceDescriptor& query, size_t top_n,
        uint64_t max_keyframe) const;

    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t descriptorDim() const { return descriptor_dim_; }

   private:
    // Keeps the FAISS headers out of this one
    struct Index;

    size_t descriptor_dim_;
    std::unique_ptr<Index> index_;
};

}  // namespace wslam::v2::mapping::loop_closure
