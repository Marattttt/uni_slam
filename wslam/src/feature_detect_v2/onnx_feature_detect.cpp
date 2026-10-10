#include "onnx_feature_detect.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <string_view>
#include <utility>
#include <vector>

#include "common.hpp"
#include "raco_lightglue.hpp"

using namespace wslam;
namespace model = wslam::v2::raco_lightglue;

#define LOG_ID "[ORT Feature Detect pass]"

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

std::string v2::OrtFeatureDetectPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::OrtFeatureDetectPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    if (auto err = loadSession()) {
        return "loading session: " + err.value();
    }

    if (auto err = validateModel()) {
        return "validating model: " + err.value();
    }

    if (auto err = initBinding()) {
        return "binding tensors: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> v2::OrtFeatureDetectPass::validateModel() const {
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
            .shape = {-1, bindings_->getKeypointCount(), 2},
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

    if (!session_) {
        return "session is not loaded";
    }
    const Ort::Session& session = *session_;

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

std::optional<std::string> v2::OrtFeatureDetectPass::initBinding() {
    if (!session_) {
        return "session is not loaded";
    }

    auto images = compute::OrtProvider::wrapBuffer(
        bindings_->getImagesBuffer(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        model::kImagesShape);
    if (!images) {
        return "images: " + images.error();
    }
    const Ort::Value& images_tensor
        = images_tensor_.emplace(std::move(images).value());

    auto keypoints = compute::OrtProvider::wrapBuffer(
        bindings_->getKeypointsBuffer(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        model::KeypointsShape(bindings_->getKeypointCount()));
    if (!keypoints) {
        return "keypoints: " + keypoints.error();
    }
    const Ort::Value& keypoints_tensor
        = keypoints_tensor_.emplace(std::move(keypoints).value());

    try {
        Ort::IoBinding& io_binding = io_binding_.emplace(*session_);

        io_binding.BindInput(model::kImagesName, images_tensor);

        // Output order here is the order GetOutputValues() returns them in
        io_binding.BindOutput(model::kKeypointsName, keypoints_tensor);

        cpu_memory_info_
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        bindDynamicOutputs(io_binding);
    } catch (const Ort::Exception& e) {
        return std::format("creating IO binding: {}", e.what());
    }

    return std::nullopt;
}

void v2::OrtFeatureDetectPass::bindDynamicOutputs(Ort::IoBinding& io_binding) {
    // A run leaves its allocated outputs in the binding, and the next run
    // would treat them as pre-allocated outputs whose shape has to match.
    // Binding by memory info again resets them to unallocated, in place
    io_binding.BindOutput(model::kMatchesName, cpu_memory_info_);
    io_binding.BindOutput(model::kMatchScoresName, cpu_memory_info_);
}

std::optional<std::string> v2::OrtFeatureDetectPass::submitWork() {
    if (!bindings_->hasPrevious()) {
        spdlog::info(LOG_ID " No previous frame to match against; skipping");
        return std::nullopt;
    }

    if (!session_ || !io_binding_) {
        return "pass is not initialized";
    }

    Ort::Session& session = *session_;
    Ort::IoBinding& io_binding = *io_binding_;

    spdlog::info(LOG_ID " Executing");

    try {
        bindDynamicOutputs(io_binding);
        session.Run(Ort::RunOptions{}, io_binding);

        auto outputs = io_binding.GetOutputValues();
        if (auto err = publishMatches(outputs)) {
            return "publishing matches: " + err.value();
        }
    } catch (const Ort::Exception& e) {
        return std::format("running model: {}", e.what());
    }

    storage_.set(ResourceIdentifier::OrtKeypointsReadyName, true);

    return std::nullopt;
}

std::optional<std::string> v2::OrtFeatureDetectPass::publishMatches(
    std::vector<Ort::Value>& outputs) {
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

    std::vector<model::KeypointMatch> match_rows(match_count);
    std::vector<float> match_scores(match_count);

    if (match_count > 0) {
        std::memcpy(match_rows.data(), matches.GetTensorData<int64_t>(),
                    match_count * sizeof(model::KeypointMatch));
        std::memcpy(match_scores.data(), scores.GetTensorData<float>(),
                    match_count * sizeof(float));
    }

    // Guards the KeypointMatch row layout
    const int64_t keypoint_count = bindings_->getKeypointCount();
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

    storage_.set(ResourceIdentifier::OrtMatchesName, std::move(match_rows));
    storage_.set(ResourceIdentifier::OrtMatchScoresName,
                 std::move(match_scores));

    return std::nullopt;
}
