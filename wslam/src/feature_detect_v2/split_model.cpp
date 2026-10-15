#include "split_model.hpp"

#include <spdlog/spdlog.h>

#include <format>
#include <span>
#include <utility>

#include "assert.hpp"
#include "common.hpp"
#include "onnx_utils.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[Split Model]"

std::optional<std::string> v2::SplitModelRunner::initialize() {
    try {
        cpu_memory_info_
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    } catch (const Ort::Exception& e) {
        return std::format("creating CPU memory info: {}", e.what());
    }

    padded_matches_.resize(static_cast<size_t>(keypoint_count_));
    padded_scores_.resize(static_cast<size_t>(keypoint_count_));

    auto extractor = loadSession(paths_.extractor);
    if (!extractor) {
        return "loading extractor: " + std::move(extractor).error();
    }
    if (auto err = ValidateExtractor(*extractor, keypoint_count_)) {
        return std::format("validating extractor {}: {}",
                           paths_.extractor.string(), err.value());
    }
    extractor_ = std::move(extractor).value();

    return std::nullopt;
}

std::expected<Ort::Session, std::string> v2::SplitModelRunner::loadSession(
    const std::filesystem::path& path) const {
    return ort_->LoadModel(path, use_webgpu_ep_);
}

std::optional<std::string> v2::SplitModelRunner::run(const Ort::Value& images) {
    if (extractor_.has_value()) {
        if (auto err = runExtractor(images)) {
            return "extractor: " + err.value();
        }
        return swapToCached();
    }

    WSLAM_ASSERT(cached_.has_value(), "running before initialize()");

    advanceReference();
    if (auto err = runCached(images)) {
        return "cached model: " + err.value();
    }
    return std::nullopt;
}

std::optional<std::string> v2::SplitModelRunner::runExtractor(
    const Ort::Value& images) {
    spdlog::info(LOG_ID " Extracting the first frame's features");

    auto features = std::make_shared<model::FrameFeatures>(
        model::FrameFeatures::Allocate(keypoint_count_));

    try {
        Ort::IoBinding io_binding{*extractor_};
        io_binding.BindInput(model::kImagesName, images);
        io_binding.BindOutput(model::kKeypointsName,
                              keypointsTensor(*features));
        io_binding.BindOutput(model::kDescriptorsName,
                              descriptorsTensor(*features));
        extractor_->Run(Ort::RunOptions{}, io_binding);
    } catch (const Ort::Exception& e) {
        return std::format("running: {}", e.what());
    }

    // The first frame is the first reference; there is nothing to match yet
    reference_ = features;
    publishFeatures(std::move(features));

    return std::nullopt;
}

std::optional<std::string> v2::SplitModelRunner::swapToCached() {
    // Released before the cached model loads, so the two are never resident
    // together
    extractor_.reset();
    spdlog::info(LOG_ID " Released the extractor; loading the cached model");

    auto cached = loadSession(paths_.cached);
    if (!cached) {
        return "loading cached model: " + std::move(cached).error();
    }
    if (auto err = ValidateCachedModel(*cached, keypoint_count_)) {
        return std::format("validating cached model {}: {}",
                           paths_.cached.string(), err.value());
    }
    cached_ = std::move(cached).value();

    return std::nullopt;
}

void v2::SplitModelRunner::advanceReference() {
    // Consumed whatever happens, so a stale marker can't advance the
    // reference later
    const bool marked
        = storage_.take<bool>(ResourceIdentifier::FeatureReferenceAdvanceName)
              .value_or(false);

    const bool advance = advance_ == ReferenceAdvance::EveryFrame || marked;
    if (!advance || pending_ == nullptr) {
        spdlog::info(LOG_ID " Keeping the reference frame");
        return;
    }

    spdlog::info(LOG_ID " Advancing the reference frame");
    reference_ = std::move(pending_);
    pending_ = nullptr;
}

std::optional<std::string> v2::SplitModelRunner::runCached(
    const Ort::Value& images) {
    WSLAM_ASSERT(reference_ != nullptr, "the extractor set the reference");

    spdlog::info(LOG_ID " Matching against the reference frame");

    auto features = std::make_shared<model::FrameFeatures>(
        model::FrameFeatures::Allocate(keypoint_count_));

    try {
        const auto matches_shape = model::MatchesShape(keypoint_count_);
        const auto scores_shape = model::MatchScoresShape(keypoint_count_);

        // A KeypointMatch is one row of the matches tensor
        const std::span match_values{
            reinterpret_cast<int64_t*>(padded_matches_.data()),
            padded_matches_.size() * model::kMatchesColumns};
        Ort::Value matches = Ort::Value::CreateTensor<int64_t>(
            cpu_memory_info_, match_values.data(), match_values.size(),
            matches_shape.data(), matches_shape.size());
        Ort::Value scores = Ort::Value::CreateTensor<float>(
            cpu_memory_info_, padded_scores_.data(), padded_scores_.size(),
            scores_shape.data(), scores_shape.size());

        Ort::IoBinding io_binding{*cached_};
        io_binding.BindInput(model::kImagesName, images);
        io_binding.BindInput(model::kCachedKeypointsName,
                             keypointsTensor(*reference_));
        io_binding.BindInput(model::kCachedDescriptorsName,
                             descriptorsTensor(*reference_));
        io_binding.BindOutput(model::kKeypointsName,
                              keypointsTensor(*features));
        io_binding.BindOutput(model::kDescriptorsName,
                              descriptorsTensor(*features));
        io_binding.BindOutput(model::kMatchesName, matches);
        io_binding.BindOutput(model::kMatchScoresName, scores);
        cached_->Run(Ort::RunOptions{}, io_binding);
    } catch (const Ort::Exception& e) {
        return std::format("running: {}", e.what());
    }

    auto parsed
        = ParseMatches(padded_matches_, padded_scores_, keypoint_count_);
    if (!parsed) {
        return "parsing matches: " + std::move(parsed).error();
    }
    PublishMatches(std::move(parsed).value(), storage_);

    std::vector<model::KeypointPosition> pair_keypoints;
    pair_keypoints.reserve(model::PairKeypointsCount(keypoint_count_));
    pair_keypoints.append_range(reference_->keypoints);
    pair_keypoints.append_range(features->keypoints);
    storage_.set(ResourceIdentifier::OrtKeypointsName,
                 std::move(pair_keypoints));

    pending_ = features;
    publishFeatures(std::move(features));
    storage_.set(ResourceIdentifier::OrtKeypointsReadyName, true);

    return std::nullopt;
}

void v2::SplitModelRunner::publishFeatures(
    std::shared_ptr<const model::FrameFeatures> features) {
    storage_.set(ResourceIdentifier::OrtFrameFeaturesName, std::move(features));
}

Ort::Value v2::SplitModelRunner::keypointsTensor(
    const model::FrameFeatures& features) const {
    const auto shape = model::KeypointsShape(keypoint_count_);
    // A KeypointPosition is two floats, one row of the keypoints tensor.
    // CreateTensor takes a mutable pointer; inputs are only read
    auto* values = const_cast<float*>(
        reinterpret_cast<const float*>(features.keypoints.data()));
    return Ort::Value::CreateTensor<float>(cpu_memory_info_, values,
                                           features.keypoints.size() * 2,
                                           shape.data(), shape.size());
}

Ort::Value v2::SplitModelRunner::descriptorsTensor(
    const model::FrameFeatures& features) const {
    const auto shape = model::DescriptorsShape(keypoint_count_);
    auto* values = const_cast<float*>(features.descriptors.data());
    return Ort::Value::CreateTensor<float>(cpu_memory_info_, values,
                                           features.descriptors.size(),
                                           shape.data(), shape.size());
}
