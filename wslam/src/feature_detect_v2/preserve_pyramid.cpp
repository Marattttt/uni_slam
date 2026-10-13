#include "preserve_pyramid.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <utility>

#include "common.hpp"

using namespace wslam;

#define LOG_ID "[Preserve Pyramid pass]"

std::string v2::PreservePyramidPass::getId() const { return LOG_ID; }

std::optional<std::string> v2::PreservePyramidPass::initialize() {
    spdlog::info(LOG_ID " initializing");

    return std::nullopt;
}

bool v2::PreservePyramidPass::shouldAdvance() {
    // Consume the marker whatever happens, so a stale one can't advance the
    // reference later
    const bool marked
        = storage_.take<bool>(ResourceIdentifier::FeatureReferenceAdvanceName)
              .value_or(false);

    const bool had_current = std::exchange(has_current_, true);
    if (!had_current) {
        return false;
    }

    switch (advance_) {
        case ReferenceAdvance::EveryFrame:
            return true;
        case ReferenceAdvance::OnKeyframeMarker:
            return marked;
    }
    return false;
}

std::optional<std::string> v2::PreservePyramidPass::submitWork() {
    if (!shouldAdvance()) {
        spdlog::info(LOG_ID " Keeping the reference frame");
        return std::nullopt;
    }

    spdlog::info(LOG_ID " Executing");

    auto record = [&](const wgpu::CommandEncoder& encoder) {
        auto copy_levels = [&] {
            for (uint32_t lod = 0; lod < GPUConst::levels_of_detail; ++lod) {
                const wgpu::Texture& current
                    = bindings_->getTexture(PyramidSlot::Current, lod);
                const wgpu::Texture& previous
                    = bindings_->getTexture(PyramidSlot::Previous, lod);

                const wgpu::TexelCopyTextureInfo source{
                    .texture = current,
                    .mipLevel = 0,
                    .origin = {.x = 0, .y = 0},
                    .aspect = wgpu::TextureAspect::All,
                };
                const wgpu::TexelCopyTextureInfo destination{
                    .texture = previous,
                    .mipLevel = 0,
                    .origin = {.x = 0, .y = 0},
                    .aspect = wgpu::TextureAspect::All,
                };
                const wgpu::Extent3D extent{
                    .width = current.GetWidth(),
                    .height = current.GetHeight(),
                    .depthOrArrayLayers = 1,
                };

                encoder.CopyTextureToTexture(&source, &destination, &extent);
            }
        };

        return gpu_->getAwaiter()
            .runChecked(copy_levels, LOG_ID " copy pyramid levels")
            .executeAll()
            .transform([](const auto& err) { return "awaiter: " + err; });
    };

    if (auto err = gpu_->submitCommands(LOG_ID " encoder", record)) {
        return err;
    }

    // The copy is submitted here, so it is queue-ordered before the fill pass
    // overwrites Current and before anything that reads Previous
    bindings_->markPreviousValid();

    return std::nullopt;
}
