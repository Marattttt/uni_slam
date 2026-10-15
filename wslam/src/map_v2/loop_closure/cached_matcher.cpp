#include "cached_matcher.hpp"

#include <array>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "feature_detect_v2/onnx_utils.hpp"

using namespace wslam::v2::mapping::loop_closure;
namespace model = wslam::v2::raco_lightglue;

std::expected<std::unique_ptr<CachedMatcher>, std::string>
CachedMatcher::Create(Ort::Session session, int64_t keypoint_count) {
    if (auto err = v2::ValidateCachedModel(session, keypoint_count)) {
        return std::unexpected("matcher model: " + std::move(err).value());
    }
    return std::unique_ptr<CachedMatcher>(
        new CachedMatcher(std::move(session), keypoint_count));
}

std::expected<PairMatches, std::string> CachedMatcher::match(
    const model::FrameFeatures& candidate, const PlaceImage& query) {
    if (candidate.keypointCount() != keypoint_count_) {
        return std::unexpected(
            std::format("candidate has {} keypoints, the model takes {}",
                        candidate.keypointCount(), keypoint_count_));
    }

    std::vector<float> images = ToMatcherInput(query);

    try {
        const auto memory_info
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const auto keypoints_shape = model::KeypointsShape(keypoint_count_);
        const auto descriptors_shape = model::DescriptorsShape(keypoint_count_);

        // Inputs are only read; CreateTensor takes mutable pointers
        auto* keypoint_values = const_cast<float*>(
            reinterpret_cast<const float*>(candidate.keypoints.data()));
        auto* descriptor_values
            = const_cast<float*>(candidate.descriptors.data());

        const std::array inputs{
            Ort::Value::CreateTensor<float>(
                memory_info, images.data(), images.size(),
                model::kImagesShape.data(), model::kImagesShape.size()),
            Ort::Value::CreateTensor<float>(
                memory_info, keypoint_values, candidate.keypoints.size() * 2,
                keypoints_shape.data(), keypoints_shape.size()),
            Ort::Value::CreateTensor<float>(
                memory_info, descriptor_values, candidate.descriptors.size(),
                descriptors_shape.data(), descriptors_shape.size()),
        };

        const std::array input_names{model::kImagesName,
                                     model::kCachedKeypointsName,
                                     model::kCachedDescriptorsName};
        const std::array output_names{model::kKeypointsName,
                                      model::kMatchesName,
                                      model::kMatchScoresName};
        auto outputs = session_.Run(Ort::RunOptions{}, input_names.data(),
                                    inputs.data(), inputs.size(),
                                    output_names.data(), output_names.size());

        const Ort::Value& keypoints = outputs.at(0);
        const Ort::Value& matches = outputs.at(1);
        const Ort::Value& scores = outputs.at(2);

        // The contract fixes every output shape, checked at Create()
        const auto rows = static_cast<size_t>(keypoint_count_);

        // A KeypointPosition is two floats and a KeypointMatch three int64,
        // one tensor row each
        const std::span query_keypoints{
            reinterpret_cast<const model::KeypointPosition*>(
                keypoints.GetTensorData<float>()),
            rows};
        const std::span match_rows{
            reinterpret_cast<const model::KeypointMatch*>(
                matches.GetTensorData<int64_t>()),
            rows};

        return PairMatches::FromMatcherOutput(
            candidate.keypoints, query_keypoints, match_rows,
            std::span{scores.GetTensorData<float>(), rows}, keypoint_count_);
    } catch (const Ort::Exception& e) {
        return std::unexpected(std::format("running matcher: {}", e.what()));
    }
}
