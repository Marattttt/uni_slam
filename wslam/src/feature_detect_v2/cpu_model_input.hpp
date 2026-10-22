#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wslam::data {
struct FrameBW;
}

namespace wslam::v2 {

// Host-side model input of the CPU execution path: one preallocated NCHW
// float buffer that SensorLoaderPass fills and OrtCpuFeatureDetectPass binds
// to its Ort::Value. Allocated once and never resized, because the tensor
// holds its address
class CpuModelInput {
   public:
    CpuModelInput();

    // Centre-crops the frame and replicates its grayscale value into the
    // three channels, exactly as resources/pack_model_input.wgsl
    [[nodiscard]] std::optional<std::string> writeFrame(
        const data::FrameBW& frame);

    [[nodiscard]] std::span<float> values() { return values_; }
    [[nodiscard]] std::span<const float> values() const { return values_; }

   private:
    std::vector<float> values_;
};

}  // namespace wslam::v2
