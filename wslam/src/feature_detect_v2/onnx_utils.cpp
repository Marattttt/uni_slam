#include "onnx_utils.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <format>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

#include "common.hpp"
#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[ORT Feature Detect]"

namespace {
struct TensorSpec {
    std::string_view name;
    ONNXTensorElementDataType element_type;
    // Dimensions the model fixes, -1 where it is dynamic
    std::vector<int64_t> shape;
};

[[nodiscard]] std::optional<std::string> CheckTensor(
    const TensorSpec& expected, std::string_view name,
    const Ort::TypeInfo& type_info) {
    if (name != expected.name) {
        return std::format("expected tensor '{}', got '{}'", expected.name,
                           name);
    }

    const auto info = type_info.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != expected.element_type) {
        return std::format("'{}' has element type {}, expected {}", name,
                           static_cast<int>(info.GetElementType()),
                           static_cast<int>(expected.element_type));
    }

    const std::vector<int64_t> shape = info.GetShape();
    if (shape.size() != expected.shape.size()) {
        return std::format("'{}' has rank {}, expected {}", name, shape.size(),
                           expected.shape.size());
    }

    for (size_t dim = 0; dim < shape.size(); ++dim) {
        if (expected.shape[dim] >= 0 && shape[dim] != expected.shape[dim]) {
            return std::format("'{}' dimension {} is {}, expected {}", name,
                               dim, shape[dim], expected.shape[dim]);
        }
    }

    return std::nullopt;
}
}  // namespace

std::optional<std::string> v2::ValidateModel(const Ort::Session& session,
                                             int64_t keypoint_count) {
    const std::array<TensorSpec, 1> inputs{
        TensorSpec{
            .name = model::kImagesName,
            .element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            .shape = {-1, model::kChannels, -1, -1},
        },
    };
    const std::array<TensorSpec, 3> outputs{
        TensorSpec{
            .name = model::kKeypointsName,
            .element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            .shape = {-1, keypoint_count, 2},
        },
        TensorSpec{
            .name = model::kMatchesName,
            .element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            .shape = {-1, model::kMatchesColumns},
        },
        TensorSpec{
            .name = model::kMatchScoresName,
            .element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            .shape = {-1},
        },
    };

    try {
        if (session.GetInputCount() != inputs.size()
            || session.GetOutputCount() != outputs.size()) {
            return std::format(
                "model has {} inputs and {} outputs, expected {} and {}",
                session.GetInputCount(), session.GetOutputCount(),
                inputs.size(), outputs.size());
        }

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto name = session.GetInputNameAllocated(i, allocator);
            if (auto err = CheckTensor(inputs.at(i), name.get(),
                                       session.GetInputTypeInfo(i))) {
                return "input: " + err.value();
            }
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto name = session.GetOutputNameAllocated(i, allocator);
            if (auto err = CheckTensor(outputs.at(i), name.get(),
                                       session.GetOutputTypeInfo(i))) {
                return "output: " + err.value();
            }
        }
    } catch (const Ort::Exception& e) {
        return std::format("querying model signature: {}", e.what());
    }

    return std::nullopt;
}

void v2::BindDynamicOutputs(Ort::IoBinding& io_binding,
                            const Ort::MemoryInfo& memory_info) {
    io_binding.BindOutput(model::kMatchesName, memory_info);
    io_binding.BindOutput(model::kMatchScoresName, memory_info);
}

std::optional<std::string> v2::PublishMatches(std::vector<Ort::Value>& outputs,
                                              int64_t keypoint_count,
                                              AnyBag& storage) {
    // Bound outputs: keypoints, matches, match scores
    constexpr size_t kMatchesIndex = 1;
    constexpr size_t kScoresIndex = 2;

    if (outputs.size() != 3) {
        return std::format("expected 3 outputs, got {}", outputs.size());
    }

    const Ort::Value& matches = outputs.at(kMatchesIndex);
    const Ort::Value& scores = outputs.at(kScoresIndex);

    const std::vector<int64_t> matches_shape
        = matches.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> scores_shape
        = scores.GetTensorTypeAndShapeInfo().GetShape();

    if (matches_shape.size() != 2
        || matches_shape[1] != model::kMatchesColumns) {
        return std::format("matches have shape {}, expected [M, {}]",
                           matches_shape, model::kMatchesColumns);
    }

    const auto match_count = static_cast<size_t>(matches_shape[0]);

    if (scores_shape.size() != 1
        || std::cmp_not_equal(scores_shape[0], match_count)) {
        return std::format("match scores have shape {}, expected [{}]",
                           scores_shape, match_count);
    }

    // Views over the tensors sized by their checked shapes, so the copies
    // below cannot read past them
    const std::span match_values{
        matches.GetTensorData<int64_t>(),
        match_count * static_cast<size_t>(model::kMatchesColumns)};
    const std::span score_values{scores.GetTensorData<float>(), match_count};

    auto match_rows = match_values | std::views::chunk(model::kMatchesColumns)
                      | std::views::transform([](const auto& row) {
                            return model::KeypointMatch{
                                .pair_index = row[0],
                                .left_index = row[1],
                                .right_index = row[2],
                            };
                        })
                      | std::ranges::to<std::vector>();
    auto match_scores = score_values | std::ranges::to<std::vector>();

    // Rejects rows that don't index the model's pairs and keypoints
    for (const model::KeypointMatch& match : match_rows) {
        const auto is_keypoint_index = [keypoint_count](int64_t index) {
            return index >= 0 && index < keypoint_count;
        };
        if (match.pair_index < 0 || match.pair_index >= model::kPairCount
            || !is_keypoint_index(match.left_index)
            || !is_keypoint_index(match.right_index)) {
            return std::format("malformed match row [{}, {}, {}]",
                               match.pair_index, match.left_index,
                               match.right_index);
        }
    }

    if (match_count > 0) {
        const auto [min_score, max_score]
            = std::ranges::minmax_element(match_scores);
        spdlog::info(LOG_ID " Got {} matches, scores in [{}, {}]", match_count,
                     *min_score, *max_score);
    } else {
        spdlog::info(LOG_ID " Got no matches");
    }

    storage.set(ResourceIdentifier::OrtMatchesName, std::move(match_rows));
    storage.set(ResourceIdentifier::OrtMatchScoresName,
                std::move(match_scores));

    return std::nullopt;
}
