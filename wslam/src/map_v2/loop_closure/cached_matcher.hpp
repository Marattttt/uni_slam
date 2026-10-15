#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include "feature_detect_v2/raco_lightglue.hpp"
#include "model_io.hpp"

namespace wslam::v2::mapping::loop_closure {

// The front end's cached RaCo-ALIKED-LightGlue model, run for geometric
// verification: it extracts the query keyframe's image and matches it against
// the candidate keyframe's features, which the front end cached when the
// candidate was its current frame. A session of its own, on the loop closure
// thread. Not thread-safe
class CachedMatcher {
   public:
    // Validates the session against the cached model contract
    [[nodiscard]] static std::expected<std::unique_ptr<CachedMatcher>,
                                       std::string>
    Create(Ort::Session session, int64_t keypoint_count);

    // Matches the candidate (left, the earlier keyframe) to the query image
    // (right)
    [[nodiscard]] std::expected<PairMatches, std::string> match(
        const raco_lightglue::FrameFeatures& candidate,
        const PlaceImage& query);

   private:
    CachedMatcher(Ort::Session session, int64_t keypoint_count)
        : session_(std::move(session)), keypoint_count_(keypoint_count) {}

    Ort::Session session_;
    const int64_t keypoint_count_;
};

}  // namespace wslam::v2::mapping::loop_closure
