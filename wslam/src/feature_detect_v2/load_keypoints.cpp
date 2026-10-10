#include "load_keypoints.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ranges>
#include <span>
#include <utility>
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

    auto read = gpu_->readBuffer<model::KeypointPosition>(
        bindings_->getKeypointsBuffer(), model::KeypointsCount(keypoint_count));
    if (!read) {
        return "reading keypoints: " + read.error();
    }
    std::vector<model::KeypointPosition> keypoints = std::move(read).value();

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
