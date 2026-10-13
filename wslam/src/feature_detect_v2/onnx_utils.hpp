#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"

// Model handling shared by the ONNX feature detect passes, whichever execution
// provider runs the model
namespace wslam::v2 {

// Checks the session's inputs and outputs against the model contract in
// raco_lightglue.hpp, for a model returning keypoint_count keypoints per image
[[nodiscard]] std::optional<std::string> ValidateModel(
    const Ort::Session& session, int64_t keypoint_count);

// Binds the data-dependent outputs (matches, scores) to memory ORT allocates
// during the run. A run leaves its allocated outputs in the binding, and the
// next run would treat them as pre-allocated outputs whose shape has to match.
// Binding by memory info again resets them to unallocated, in place, so call
// it before every run
void BindDynamicOutputs(Ort::IoBinding& io_binding,
                        const Ort::MemoryInfo& memory_info);

// Checks the matches and scores of a run's bound outputs (keypoints, matches,
// match scores, in that order) and publishes them to storage under
// ResourceIdentifier::OrtMatchesName and OrtMatchScoresName
[[nodiscard]] std::optional<std::string> PublishMatches(
    std::vector<Ort::Value>& outputs, int64_t keypoint_count, AnyBag& storage);

}  // namespace wslam::v2
