#include "feature_detect.hpp"

#include "anybag.hpp"
#include "common.hpp"
#include "detect_corners.hpp"
#include "fill_pyramid.hpp"
#include "provider_base.hpp"
#include "sensor_loader.hpp"

using namespace wslam;

class StorageImageProvider {
   public:
    StorageImageProvider(AnyBag& storage) : storage_(storage) {}

    std::optional<std::span<const std::byte>> operator()() {
        auto frame
            = storage_.get<data::FrameBW>(ResourceIdentifier::GetFrameName(0));
        if (!frame) {
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
    std::generator<std::expected<data::Reading<1>, std::string>> provider) {
    const auto gpu = compute.getGPUPtr();

    compute::Stage stage{"Feature detect", gpu};

    StorageImageProvider img_provider{compute.getStorage()};

    stage.add_pass(std::make_unique<SensorLoaderPass>(std::move(provider), gpu,
                                                      compute.getStorage()));
    stage.add_pass(std::make_unique<FillPyramidPass>(
        gpu, shared_bindings,
        PassFillPyramidOpts{.image_getter = std::move(img_provider)}));
    stage.add_pass(std::make_unique<PassDetectCorners>(gpu, shared_bindings));

    return stage;
}
