#include "pair_matcher.hpp"

#include <array>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "feature_detect_v2/onnx_utils.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"

using namespace wslam::v2::mapping::loop_closure;
namespace model = wslam::v2::raco_lightglue;

std::expected<std::unique_ptr<PairMatcher>, std::string> PairMatcher::Create(
    Ort::Session session, int64_t keypoint_count) {
    if (auto err = v2::ValidateModel(session, keypoint_count)) {
        return std::unexpected("matcher model: " + std::move(err).value());
    }
    return std::unique_ptr<PairMatcher>(
        new PairMatcher(std::move(session), keypoint_count));
}

std::expected<PairMatches, std::string> PairMatcher::match(
    const PlaceImage& left, const PlaceImage& right) {
    std::vector<float> images = ToMatcherInput(left, right);

    try {
        const auto memory_info
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const Ort::Value images_tensor = Ort::Value::CreateTensor<float>(
            memory_info, images.data(), images.size(),
            model::kImagesShape.data(), model::kImagesShape.size());

        const std::array input_names{model::kImagesName};
        const std::array output_names{model::kKeypointsName,
                                      model::kMatchesName,
                                      model::kMatchScoresName};
        auto outputs = session_.Run(Ort::RunOptions{}, input_names.data(),
                                    &images_tensor, 1, output_names.data(),
                                    output_names.size());

        const Ort::Value& keypoints = outputs.at(0);
        const Ort::Value& matches = outputs.at(1);
        const Ort::Value& scores = outputs.at(2);

        const size_t keypoint_values
            = keypoints.GetTensorTypeAndShapeInfo().GetElementCount();
        const std::vector<int64_t> matches_shape
            = matches.GetTensorTypeAndShapeInfo().GetShape();
        const size_t score_count
            = scores.GetTensorTypeAndShapeInfo().GetElementCount();

        if (matches_shape.size() != 2
            || matches_shape[1] != model::kMatchesColumns) {
            return std::unexpected(
                std::format("matches have shape {}, expected [M, {}]",
                            matches_shape, model::kMatchesColumns));
        }
        const auto match_count = static_cast<size_t>(matches_shape[0]);

        // A KeypointPosition is two floats and a KeypointMatch three int64,
        // one tensor row each
        const std::span keypoint_rows{
            reinterpret_cast<const model::KeypointPosition*>(
                keypoints.GetTensorData<float>()),
            keypoint_values / 2};
        const std::span match_rows{
            reinterpret_cast<const model::KeypointMatch*>(
                matches.GetTensorData<int64_t>()),
            match_count};

        return PairMatches::FromMatcherOutput(
            keypoint_rows, match_rows,
            std::span{scores.GetTensorData<float>(), score_count},
            keypoint_count_);
    } catch (const Ort::Exception& e) {
        return std::unexpected(std::format("running matcher: {}", e.what()));
    }
}
