#pragma once

#include <memory>

#include "common.hpp"
#include "gpu.hpp"
#include "models.hpp"
#include "pass.hpp"

namespace wslam {
class LoadDataCPUPass : public compute::Pass {
   public:
    LoadDataCPUPass(GpuSharedBindings& shared,
                    std::shared_ptr<compute::GPU> gpu,
                    std::string textures_out_label,
                    std::pair<std::string, std::string> features)
        : compute::Pass(std::move(gpu)),
          kTexturesOutputLabel(std::move(textures_out_label)),
          kFeaturesGPULabel(std::move(features.first)),
          kFeaturesOutputLabel(std::move(features.second)),
          shared_(shared) {}

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    static constexpr size_t kLodCount = GPUConst::levels_of_detail;

    const std::string kTexturesGPULabel;
    const std::string kTexturesOutputLabel;
    const std::string kFeaturesGPULabel;
    const std::string kFeaturesOutputLabel;

    GpuSharedBindings& shared_;

    std::optional<compute::BufferBinding> features_binding_;
    std::optional<std::array<wgpu::Texture, kLodCount>> textures_;

    std::optional<std::string> initBindings();

    std::expected<std::array<compute::TextureData, kLodCount>, std::string>
    loadTextureData();
    std::expected<std::vector<Feature>, std::string> loadFeatures();
};
};  // namespace wslam
