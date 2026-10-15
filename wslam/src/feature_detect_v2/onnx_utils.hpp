#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "raco_lightglue.hpp"

// Model handling shared by the ONNX feature detect passes and the loop
// closure matcher, whichever execution provider runs the model
namespace wslam::v2 {

// Check the session's inputs and outputs against the extractor and the cached
// model contracts in raco_lightglue.hpp, for a model returning keypoint_count
// keypoints per image
[[nodiscard]] std::optional<std::string> ValidateExtractor(
    const Ort::Session& session, int64_t keypoint_count);
[[nodiscard]] std::optional<std::string> ValidateCachedModel(
    const Ort::Session& session, int64_t keypoint_count);

// The real matches of a cached model run
struct ParsedMatches {
    std::vector<raco_lightglue::KeypointMatch> matches;
    std::vector<float> scores;
};

// Keeps the rows of the cached model's zero-padded matches (keypoint_count
// rows) whose score is positive. Fails on a kept row that doesn't index the
// keypoints of both images
[[nodiscard]] std::expected<ParsedMatches, std::string> ParseMatches(
    std::span<const raco_lightglue::KeypointMatch> padded_matches,
    std::span<const float> padded_scores, int64_t keypoint_count);

// Publishes parsed matches to storage under ResourceIdentifier::OrtMatchesName
// and OrtMatchScoresName
void PublishMatches(ParsedMatches parsed, AnyBag& storage);

}  // namespace wslam::v2
