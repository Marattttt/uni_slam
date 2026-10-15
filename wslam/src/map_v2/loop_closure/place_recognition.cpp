#include "place_recognition.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <format>
#include <span>
#include <utility>

using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;

#define LOG_ID "[Place Recognition]"

namespace {
struct AllowedInput {
    uint32_t width;
    uint32_t height;
};

// Sizes the exports were made at, width x height
constexpr std::array kMegaLocInputs{
    AllowedInput{.width = 322, .height = 224},
    AllowedInput{.width = 322, .height = 322},
};
constexpr std::array kMixVprInputs{
    AllowedInput{.width = 320, .height = 320},
};

[[nodiscard]] std::expected<std::vector<int64_t>, std::string> FloatShape(
    const Ort::TypeInfo& type_info, std::string_view what) {
    const auto info = type_info.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return std::unexpected(
            std::format("{} has element type {}, expected float32", what,
                        static_cast<int>(info.GetElementType())));
    }
    return info.GetShape();
}
}  // namespace

std::expected<VprModelSpec, std::string> loop_closure::ReadVprModelSpec(
    VprModelKind kind, const Ort::Session& session) {
    try {
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            return std::unexpected(
                std::format("{} model has {} inputs and {} outputs, expected 1 "
                            "and 1",
                            ToString(kind), session.GetInputCount(),
                            session.GetOutputCount()));
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        if (std::string_view{input_name.get()} != kVprInputName
            || std::string_view{output_name.get()} != kVprOutputName) {
            return std::unexpected(std::format(
                "{} model tensors are '{}' -> '{}', expected '{}' -> '{}'",
                ToString(kind), input_name.get(), output_name.get(),
                kVprInputName, kVprOutputName));
        }

        auto input_shape = FloatShape(session.GetInputTypeInfo(0), "input");
        auto output_shape = FloatShape(session.GetOutputTypeInfo(0), "output");
        if (!input_shape) {
            return std::unexpected(std::move(input_shape).error());
        }
        if (!output_shape) {
            return std::unexpected(std::move(output_shape).error());
        }

        if (input_shape->size() != 4 || (*input_shape)[0] != 1
            || (*input_shape)[1] != 3) {
            return std::unexpected(
                std::format("{} input has shape {}, expected [1, 3, H, W]",
                            ToString(kind), *input_shape));
        }

        const VprModelSpec spec{
            .kind = kind,
            .input = {.width = static_cast<uint32_t>((*input_shape)[3]),
                      .height = static_cast<uint32_t>((*input_shape)[2])},
            .descriptor_dim = kind == VprModelKind::MegaLoc
                                  ? kMegaLocDescriptorDim
                                  : kMixVprDescriptorDim,
        };

        const std::span<const AllowedInput> allowed
            = kind == VprModelKind::MegaLoc
                  ? std::span<const AllowedInput>{kMegaLocInputs}
                  : std::span<const AllowedInput>{kMixVprInputs};
        const bool size_allowed
            = std::ranges::any_of(allowed, [&](const AllowedInput& size) {
                  return size.width == spec.input.width
                         && size.height == spec.input.height;
              });
        if (!size_allowed) {
            return std::unexpected(
                std::format("{} input is {}x{} (WxH), not an exported size",
                            ToString(kind), spec.input.width,
                            spec.input.height));
        }

        const std::vector<int64_t> expected_output{
            1, static_cast<int64_t>(spec.descriptor_dim)};
        if (*output_shape != expected_output) {
            return std::unexpected(
                std::format("{} output has shape {}, expected {}",
                            ToString(kind), *output_shape, expected_output));
        }

        return spec;
    } catch (const Ort::Exception& e) {
        return std::unexpected(
            std::format("querying model signature: {}", e.what()));
    }
}

std::expected<std::unique_ptr<PlaceRecognitionModel>, std::string>
PlaceRecognitionModel::Create(VprModelKind kind, Ort::Session session) {
    auto spec = ReadVprModelSpec(kind, session);
    if (!spec) {
        return std::unexpected(std::move(spec).error());
    }

    spdlog::info(LOG_ID " {} model: input {}x{} (WxH), descriptor {}-d",
                 ToString(kind), spec->input.width, spec->input.height,
                 spec->descriptor_dim);

    return std::unique_ptr<PlaceRecognitionModel>(
        new PlaceRecognitionModel(spec.value(), std::move(session)));
}

std::expected<PlaceDescriptor, std::string> PlaceRecognitionModel::describe(
    const PlaceImage& image) {
    std::vector<float> input = ToVprInput(image, spec_.input);

    const std::array<int64_t, 4> input_shape{
        1, 3, static_cast<int64_t>(spec_.input.height),
        static_cast<int64_t>(spec_.input.width)};

    try {
        const auto memory_info
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_shape.data(),
            input_shape.size());

        const std::array input_names{kVprInputName};
        const std::array output_names{kVprOutputName};
        auto outputs
            = session_.Run(Ort::RunOptions{}, input_names.data(),
                           &input_tensor, 1, output_names.data(), 1);

        const Ort::Value& output = outputs.at(0);
        const size_t count
            = output.GetTensorTypeAndShapeInfo().GetElementCount();
        return PlaceDescriptor::FromVprOutput(
            std::span{output.GetTensorData<float>(), count},
            spec_.descriptor_dim);
    } catch (const Ort::Exception& e) {
        return std::unexpected(std::format("running {}: {}",
                                           ToString(spec_.kind), e.what()));
    }
}
