#pragma once

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>

#include <expected>
#include <functional>
#include <string>

#include "provider_base.hpp"

namespace wslam::viz {
struct VizTexture {
    uint32_t width;
    uint32_t height;
    std::vector<data::PixelRGB> data;
};

struct VizFeature {
    float x;
    float y;
};

class VizGUI {
    VizGUI(std::unique_ptr<pangolin::OpenGlRenderState> state_cam,
           std::unique_ptr<pangolin::Handler3D> handler,
           pangolin::View* display_cam)
        : state_cam_(std::move(state_cam)),
          handler_(std::move(handler)),
          display_cam_(display_cam) {}

   public:
    VizGUI(VizGUI&& gui) noexcept = default;
    VizGUI& operator=(VizGUI&& gui) noexcept = default;

    struct VizInitOpts {};

    static inline const std::string kWindowTitle = "SLAM with WebGPU";

    static std::expected<VizGUI, std::string> create(VizInitOpts opts);

    bool windowShouldClose();
    void closeWindow();
    void startFrame();
    void endFrame();
    void addRequestNextCallback(std::function<void()> callback);

    struct Features {
        const std::vector<VizFeature>& features;
        std::array<uint8_t, 3> color;
        float radius;
        uint32_t image_width;
        uint32_t image_height;
    };
    void drawFeatures(Features features);

    void drawTestCube();
    void drawTexture(const VizTexture& texture);

   private:
    std::unique_ptr<pangolin::OpenGlRenderState> state_cam_;
    std::unique_ptr<pangolin::Handler3D> handler_;
    pangolin::View* display_cam_;
};
};  // namespace wslam::viz
