#include "keyframe_schedule.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <utility>

#include "assert.hpp"

using namespace wslam;

namespace {
std::expected<std::vector<uint64_t>, std::string> ReadIncreasingIndices(
    const nlohmann::json& sequence, const std::string& key) {
    const auto entry = sequence.find(key);
    if (entry == sequence.end() || !entry->is_array()) {
        return std::unexpected(std::format("'{}' is not an array", key));
    }

    std::vector<uint64_t> values;
    values.reserve(entry->size());
    for (const nlohmann::json& value : *entry) {
        // Unsigned only: get<uint64_t>() would wrap a negative number
        if (!value.is_number_unsigned()) {
            return std::unexpected(
                std::format("'{}' holds {}, expected unsigned integers", key,
                            value.dump()));
        }
        values.push_back(value.get<uint64_t>());
    }

    if (values.empty()) {
        return std::unexpected(std::format("'{}' is empty", key));
    }

    const auto unordered
        = std::ranges::adjacent_find(values, std::greater_equal{});
    if (unordered != values.end()) {
        return std::unexpected(
            std::format("'{}' is not strictly increasing at {} -> {}", key,
                        *unordered, *std::next(unordered)));
    }

    return values;
}

class KeyframeFilter {
   public:
    explicit KeyframeFilter(KeyframeSchedule schedule)
        : schedule_(
              std::make_shared<const KeyframeSchedule>(std::move(schedule))) {
        WSLAM_ASSERT(
            schedule_->frame_indices.size() == schedule_->timestamps_ns.size(),
            "schedule has {} indices and {} timestamps",
            schedule_->frame_indices.size(), schedule_->timestamps_ns.size());
    }

    std::expected<data::ReadingDecision, std::string> operator()(
        const data::ReadingCandidate& candidate) {
        using data::ReadingDecision;

        if (candidate.kind == data::ReadingCandidate::Kind::Imu) {
            return ReadingDecision::Keep;
        }

        if (next_ == schedule_->timestamps_ns.size()) {
            return ReadingDecision::StopStream;
        }

        const uint64_t keyframe_timestamp = schedule_->timestamps_ns[next_];
        const uint64_t keyframe_index = schedule_->frame_indices[next_];

        if (candidate.timestamp < keyframe_timestamp) {
            return ReadingDecision::Skip;
        }

        if (candidate.timestamp > keyframe_timestamp) {
            return std::unexpected(std::format(
                "keyframe {} of sequence '{}' (frame {}, ts {}) is not in the "
                "stream, which reached frame {} at ts {}",
                next_, schedule_->sequence, keyframe_index, keyframe_timestamp,
                candidate.index, candidate.timestamp));
        }

        if (candidate.index != keyframe_index) {
            return std::unexpected(std::format(
                "keyframe {} of sequence '{}' at ts {} is frame {} of the "
                "stream, but the schedule says frame {}",
                next_, schedule_->sequence, keyframe_timestamp, candidate.index,
                keyframe_index));
        }

        ++next_;
        return ReadingDecision::Keep;
    }

   private:
    // Shared so copies the std::function makes don't duplicate the lists
    std::shared_ptr<const KeyframeSchedule> schedule_;
    // Schedule position of the next keyframe to find
    size_t next_ = 0;
};
}  // namespace

std::expected<KeyframeSchedule, std::string> wslam::LoadKeyframeSchedule(
    const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        return std::unexpected(
            std::format("could not open '{}'", path.string()));
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(file);
    } catch (const nlohmann::json::exception& e) {
        return std::unexpected(
            std::format("parsing '{}': {}", path.string(), e.what()));
    }

    if (!json.is_object() || json.size() != 1) {
        return std::unexpected(
            std::format("'{}' must be an object holding exactly one sequence",
                        path.string()));
    }

    const auto entry = json.begin();
    if (!entry->is_object()) {
        return std::unexpected(
            std::format("sequence '{}' is not an object", entry.key()));
    }

    KeyframeSchedule schedule;
    schedule.sequence = entry.key();

    auto indices = ReadIncreasingIndices(*entry, "keyframes");
    if (!indices) {
        return std::unexpected(std::format("sequence '{}': {}", entry.key(),
                                           std::move(indices).error()));
    }
    schedule.frame_indices = std::move(indices).value();

    auto timestamps = ReadIncreasingIndices(*entry, "timestamps_ns");
    if (!timestamps) {
        return std::unexpected(std::format("sequence '{}': {}", entry.key(),
                                           std::move(timestamps).error()));
    }
    schedule.timestamps_ns = std::move(timestamps).value();

    if (schedule.frame_indices.size() != schedule.timestamps_ns.size()) {
        return std::unexpected(std::format(
            "sequence '{}' has {} keyframes but {} timestamps", entry.key(),
            schedule.frame_indices.size(), schedule.timestamps_ns.size()));
    }

    return schedule;
}

data::ReadingFilter wslam::MakeKeyframeFilter(KeyframeSchedule schedule) {
    return KeyframeFilter{std::move(schedule)};
}
