#pragma once

#include <memory>

#include "common.hpp"
#include "compute/gpu.hpp"
#include "models.hpp"
#include "compute/pass.hpp"

namespace wslam {
class LoadDataCPUPass : public compute::Pass {
   public:
    // `readback_textures` controls whether the LoD pyramid textures are
    // copied back to the CPU each frame. Their only consumer is the GUI
    // resource provider, so headless pipelines pass false and skip six
    // full GPU→CPU round-trips per frame. Feature readback always runs —
    // the matcher depends on it.
    LoadDataCPUPass(GpuSharedBindings& shared,
                    std::shared_ptr<compute::GPU> gpu, std::string features,
                    bool readback_textures)
        : kFeaturesGPULabel(std::move(features)),
          gpu_(std::move(gpu)),
          shared_(shared),
          readback_textures_(readback_textures) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    static constexpr size_t kLodCount = GPUConst::levels_of_detail;

    const std::string kFeaturesGPULabel;

    std::shared_ptr<compute::GPU> gpu_;
    GpuSharedBindings& shared_;
    bool readback_textures_;

    std::optional<compute::BufferBinding*> features_binding_;
    std::optional<std::array<wgpu::Texture, kLodCount>> textures_;

    // Frames the current reference feature set (FeatureSet(1)) has been
    // held for. Diagnostic only — long holds mean the keyframe gate is
    // rejecting everything and the matcher is being asked to bridge an
    // ever-growing baseline.
    uint64_t reference_age_frames_ = 0;

    std::optional<std::string> initBindings();

    std::expected<std::array<compute::TextureData, kLodCount>, std::string>
    loadTextureData();
    std::expected<FeatureSet, std::string> loadFeatures();

    void shiftFeatureSets();
};
};  // namespace wslam
