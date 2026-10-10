#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "data/provider_base.hpp"

namespace wslam {

// Frames chosen as keyframes in one dataset sequence, read from a JSON file
// shaped as
//   { "<sequence>": { "keyframes": [<frame index>, ...],
//                     "timestamps_ns": [<frame timestamp>, ...] } }
// Frame indices are 0-based positions in the sequence's frame stream. Both
// lists are strictly increasing and the same length
struct KeyframeSchedule {
    std::string sequence;
    std::vector<uint64_t> frame_indices;
    std::vector<uint64_t> timestamps_ns;
};

// Loads a schedule file holding exactly one sequence
[[nodiscard]] std::expected<KeyframeSchedule, std::string> LoadKeyframeSchedule(
    const std::filesystem::path& path);

// Reading filter that keeps every IMU sample and only the schedule's frames.
// A frame is matched by timestamp, and its index in the stream must equal the
// schedule's. A keyframe the stream passes without containing it is an error.
// The first frame after the last keyframe ends the stream
[[nodiscard]] data::ReadingFilter MakeKeyframeFilter(KeyframeSchedule schedule);

}  // namespace wslam
