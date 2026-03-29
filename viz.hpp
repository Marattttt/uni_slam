#pragma once

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>

#include <expected>
#include <functional>
#include <string>

namespace wslam {

struct VizTexture {
    uint32_t width;
    uint32_t height;
    uint8_t channels;
    std::vector<uint8_t> data;
};

class Viz {
    Viz(std::unique_ptr<pangolin::OpenGlRenderState> state_cam,
        std::unique_ptr<pangolin::Handler3D> handler,
        pangolin::View* display_cam)
        : state_cam_(std::move(state_cam)),
          handler_(std::move(handler)),
          display_cam_(display_cam) {}

   public:
    struct VizInitOpts {};

    static inline const std::string kWindowTitle = "SLAM with WebGPU";

    static std::expected<Viz, std::string> initialize(VizInitOpts opts);

    bool windowShouldClose();
    void closeWindow();
    void startFrame();
    void endFrame();
    void addRequestNextCallback(std::function<void()> callback);

    void drawTestCube();
    void drawTexture(const VizTexture& texture);

   private:
    std::unique_ptr<pangolin::OpenGlRenderState> state_cam_;
    std::unique_ptr<pangolin::Handler3D> handler_;
    pangolin::View* display_cam_;
};
};  // namespace wslam
