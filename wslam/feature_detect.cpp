#include "feature_detect.hpp"

#include <memory>

#include "anybag.hpp"
#include "common.hpp"
#include "cull_corners.hpp"
#include "detect_corners.hpp"
#include "fill_pyramid.hpp"
#include "generate_features.hpp"
#include "provider_base.hpp"
#include "sensor_loader.hpp"
#include "vizualize_data.hpp"

using namespace wslam;

#define LOG_ID "[Feature Detect stage]"

class StorageImageProvider {
   public:
    StorageImageProvider(AnyBag& storage) : storage_(storage) {}

    std::optional<std::span<const std::byte>> operator()() {
        const auto name = ResourceIdentifier::GetFrameName(0);

        auto frame = storage_.get<data::FrameBW>(name);
        if (!frame) {
            spdlog::warn(
                LOG_ID " could not get frame from storage by name:'{}'", name);
            return std::nullopt;
        }

        last_frame_ = std::move(frame.value());

        return std::as_bytes(std::span{last_frame_.pixels});
    }

   private:
    AnyBag& storage_;
    data::FrameBW last_frame_;
};

compute::Stage wslam::CreateFeatureDetectStage(
    compute::Compute& compute, GpuSharedBindings& shared_bindings,
    std::generator<std::expected<data::Reading<1>, std::string>> provider,
    std::string feature_output_label) {
    const auto gpu = compute.getGPUPtr();
    compute::Stage stage{"Feature detect", gpu};

    stage.add_pass(std::make_unique<SensorLoaderPass>(std::move(provider), gpu,
                                                      compute.getStorage()));

    StorageImageProvider img_provider{compute.getStorage()};
    stage.add_pass(std::make_unique<FillPyramidPass>(
        gpu, shared_bindings,
        PassFillPyramidOpts{.image_getter = std::move(img_provider),
                            .storage = shared_bindings.getStorage()}));

    stage.add_pass(
        std::make_unique<PassDetectCorners>(gpu, shared_bindings, "corners"));

    stage.add_pass(std::make_unique<CullCornersPass>(gpu, shared_bindings,
                                                     "corners", "corners_out"));

    stage.add_pass(std::make_unique<GenerateFeaturesPass>(
        gpu, shared_bindings, "corners_out", feature_output_label));

    // std::unique_ptr<viz::ResourceProvider> resource_provider
    //     = std::make_unique<viz::WgpuResourceProvider>(
    //         viz::WgpuResourceProvider::Opts{
    //             .storage = compute.getStorage(),
    //             .shared = shared_bindings,
    //             .gpu = gpu,
    //             .lod_levels = {{0}, {1}, {2}, {3}, {4}},
    //             .corners_label = "corners_out",
    //             .features_label = feature_output_label,
    //         });
    // stage.add_pass(std::make_unique<viz::VisualizeDataPass>(
    //     gpu, std::move(resource_provider)));

    return stage;
}
