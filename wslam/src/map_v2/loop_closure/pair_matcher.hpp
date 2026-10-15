#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include "model_io.hpp"

namespace wslam::v2::mapping::loop_closure {

// The front end's RaCo-ALIKED-LightGlue pipeline, run on one keyframe pair
// for geometric verification. A session of its own, on the loop closure
// thread. Not thread-safe
class PairMatcher {
   public:
    // Validates the session against the raco_lightglue contract
    [[nodiscard]] static std::expected<std::unique_ptr<PairMatcher>,
                                       std::string>
    Create(Ort::Session session, int64_t keypoint_count);

    // Matches left (the earlier keyframe) to right (the query)
    [[nodiscard]] std::expected<PairMatches, std::string> match(
        const PlaceImage& left, const PlaceImage& right);

   private:
    PairMatcher(Ort::Session session, int64_t keypoint_count)
        : session_(std::move(session)), keypoint_count_(keypoint_count) {}

    Ort::Session session_;
    const int64_t keypoint_count_;
};

}  // namespace wslam::v2::mapping::loop_closure
