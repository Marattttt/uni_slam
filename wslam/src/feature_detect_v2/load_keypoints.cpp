#include "load_keypoints.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <ranges>
#include <span>
#include <vector>

#include "common.hpp"
#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[Load Keypoints pass]"

std::string v2::LoadKeypointsPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::LoadKeypointsPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    return std::nullopt;
}

std::optional<std::string> v2::LoadKeypointsPass::execute() {
    // Set by the detect pass only on frames the model ran on
    if (!storage_.take<bool>(ResourceIdentifier::OrtKeypointsReadyName)) {
        spdlog::info(LOG_ID " No new keypoints; skipping");
        return std::nullopt;
    }

    spdlog::info(LOG_ID " Executing");

    const int64_t keypoint_count = bindings_->getKeypointCount();
    const size_t keypoints_bytes = model::KeypointsBytes(keypoint_count);

    auto bytes
        = gpu_->readBuffer(bindings_->getKeypointsBuffer(), keypoints_bytes);
    if (!bytes) {
        return "reading keypoints: " + bytes.error();
    }

    if (bytes->size() != keypoints_bytes) {
        return std::format("read {} keypoint bytes, expected {}", bytes->size(),
                           keypoints_bytes);
    }

    std::vector<model::KeypointPosition> keypoints(
        model::KeypointsCount(keypoint_count));
    std::memcpy(keypoints.data(), bytes->data(), keypoints_bytes);

    // Keypoints are stored image by image, keypoint_count each
    const auto per_image = static_cast<size_t>(keypoint_count);
    for (const auto slot : {model::kLeft, model::kRight}) {
        const auto image_keypoints = std::span{keypoints}.subspan(
            static_cast<size_t>(slot) * per_image, per_image);

        const auto [min_x, max_x] = std::ranges::minmax(
            image_keypoints | std::views::transform(&model::KeypointPosition::x));
        const auto [min_y, max_y] = std::ranges::minmax(
            image_keypoints | std::views::transform(&model::KeypointPosition::y));

        spdlog::info(
            LOG_ID " {} image: {} keypoints, x in [{}, {}], y in [{}, {}]",
            slot == model::kLeft ? "Left (previous)" : "Right (current)",
            image_keypoints.size(), min_x, max_x, min_y, max_y);
    }

    const auto matches = storage_.getPtr<std::vector<model::KeypointMatch>>(
        ResourceIdentifier::OrtMatchesName);
    if (matches) {
        spdlog::info(LOG_ID " {} matches between the two images",
                     matches.value()->size());
    }

    storage_.set(ResourceIdentifier::OrtKeypointsName, std::move(keypoints));

    return std::nullopt;
}
