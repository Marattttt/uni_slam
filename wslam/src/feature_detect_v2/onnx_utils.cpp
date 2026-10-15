#include "onnx_utils.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <format>
#include <string_view>
#include <utility>

#include "common.hpp"

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

constexpr auto kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto kInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

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

[[nodiscard]] std::optional<std::string> CheckSignature(
    const Ort::Session& session, std::span<const TensorSpec> inputs,
    std::span<const TensorSpec> outputs) {
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
            if (auto err = CheckTensor(inputs[i], name.get(),
                                       session.GetInputTypeInfo(i))) {
                return "input: " + err.value();
            }
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto name = session.GetOutputNameAllocated(i, allocator);
            if (auto err = CheckTensor(outputs[i], name.get(),
                                       session.GetOutputTypeInfo(i))) {
                return "output: " + err.value();
            }
        }
    } catch (const Ort::Exception& e) {
        return std::format("querying model signature: {}", e.what());
    }

    return std::nullopt;
}

// The image dimensions are dynamic in the exports (multiples of 32)
TensorSpec ImagesSpec() {
    return {
        .name = model::kImagesName,
        .element_type = kFloat,
        .shape = {model::kBatchSize, model::kChannels, -1, -1},
    };
}

TensorSpec KeypointsSpec(std::string_view name, int64_t keypoint_count) {
    return {
        .name = name,
        .element_type = kFloat,
        .shape = {model::kBatchSize, keypoint_count, 2},
    };
}

TensorSpec DescriptorsSpec(std::string_view name, int64_t keypoint_count) {
    return {
        .name = name,
        .element_type = kFloat,
        .shape = {model::kBatchSize, keypoint_count, model::kDescriptorDim},
    };
}
}  // namespace

std::optional<std::string> v2::ValidateExtractor(const Ort::Session& session,
                                                 int64_t keypoint_count) {
    const std::array inputs{ImagesSpec()};
    const std::array outputs{
        KeypointsSpec(model::kKeypointsName, keypoint_count),
        DescriptorsSpec(model::kDescriptorsName, keypoint_count),
    };
    return CheckSignature(session, inputs, outputs);
}

std::optional<std::string> v2::ValidateCachedModel(const Ort::Session& session,
                                                   int64_t keypoint_count) {
    const std::array inputs{
        ImagesSpec(),
        KeypointsSpec(model::kCachedKeypointsName, keypoint_count),
        DescriptorsSpec(model::kCachedDescriptorsName, keypoint_count),
    };
    const std::array outputs{
        KeypointsSpec(model::kKeypointsName, keypoint_count),
        DescriptorsSpec(model::kDescriptorsName, keypoint_count),
        TensorSpec{
            .name = model::kMatchesName,
            .element_type = kInt64,
            .shape = {keypoint_count, model::kMatchesColumns},
        },
        TensorSpec{
            .name = model::kMatchScoresName,
            .element_type = kFloat,
            .shape = {keypoint_count},
        },
    };
    return CheckSignature(session, inputs, outputs);
}

std::expected<v2::ParsedMatches, std::string> v2::ParseMatches(
    std::span<const model::KeypointMatch> padded_matches,
    std::span<const float> padded_scores, int64_t keypoint_count) {
    if (padded_matches.size() != padded_scores.size()) {
        return std::unexpected(std::format("{} match rows but {} scores",
                                           padded_matches.size(),
                                           padded_scores.size()));
    }

    const auto is_keypoint_index = [keypoint_count](int64_t index) {
        return index >= 0 && index < keypoint_count;
    };

    ParsedMatches parsed;
    for (size_t row = 0; row < padded_matches.size(); ++row) {
        // Padding rows are (0, 0, 0) with score 0, which would pass the
        // index check below
        if (!(padded_scores[row] > 0.0F)) {
            continue;
        }

        const model::KeypointMatch& match = padded_matches[row];
        if (match.pair_index != 0 || !is_keypoint_index(match.left_index)
            || !is_keypoint_index(match.right_index)) {
            return std::unexpected(std::format(
                "malformed match row {}: [{}, {}, {}]", row, match.pair_index,
                match.left_index, match.right_index));
        }

        parsed.matches.push_back(match);
        parsed.scores.push_back(padded_scores[row]);
    }

    return parsed;
}

void v2::PublishMatches(ParsedMatches parsed, AnyBag& storage) {
    if (!parsed.matches.empty()) {
        const auto [min_score, max_score]
            = std::ranges::minmax_element(parsed.scores);
        spdlog::info(LOG_ID " Got {} matches, scores in [{}, {}]",
                     parsed.matches.size(), *min_score, *max_score);
    } else {
        spdlog::info(LOG_ID " Got no matches");
    }

    storage.set(ResourceIdentifier::OrtMatchesName, std::move(parsed.matches));
    storage.set(ResourceIdentifier::OrtMatchScoresName,
                std::move(parsed.scores));
}
