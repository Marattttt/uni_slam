#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstddef>
#include <expected>
#include <memory>
#include <string>

#include "config.hpp"
#include "model_io.hpp"

// Global-descriptor place recognition models (ONNX exports of
// ~/dev/test/ort-export): one image in, one L2-normalised descriptor out.
// Both have fully static shapes:
//   MegaLoc (DINOv2 + optimal-transport aggregation, mixed fp16/fp32):
//     image [1, 3, 224, 322] or [1, 3, 322, 322] -> descriptor [1, 8448]
//   MixVPR (ResNet-50 + MixVPR, fp32):
//     image [1, 3, 320, 320] -> descriptor [1, 4096]
namespace wslam::v2::mapping::loop_closure {

constexpr const char* kVprInputName = "image";
constexpr const char* kVprOutputName = "descriptor";

constexpr size_t kMegaLocDescriptorDim = 8448;
constexpr size_t kMixVprDescriptorDim = 4096;

struct VprModelSpec {
    VprModelKind kind = VprModelKind::MixVpr;
    VprInputSpec input;
    size_t descriptor_dim = 0;
};

// Checks the session's signature against the kind's contract and returns
// the spec it has. MegaLoc accepts either exported size
[[nodiscard]] std::expected<VprModelSpec, std::string> ReadVprModelSpec(
    VprModelKind kind, const Ort::Session& session);

// Owns a place recognition session. Not thread-safe: one thread (the loop
// closure worker) runs it
class PlaceRecognitionModel {
   public:
    [[nodiscard]] static std::expected<std::unique_ptr<PlaceRecognitionModel>,
                                       std::string>
    Create(VprModelKind kind, Ort::Session session);

    [[nodiscard]] const VprModelSpec& spec() const { return spec_; }

    [[nodiscard]] std::expected<PlaceDescriptor, std::string> describe(
        const PlaceImage& image);

   private:
    PlaceRecognitionModel(VprModelSpec spec, Ort::Session session)
        : spec_(spec), session_(std::move(session)) {}

    const VprModelSpec spec_;
    Ort::Session session_;
};

}  // namespace wslam::v2::mapping::loop_closure
