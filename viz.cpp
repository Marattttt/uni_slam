#include "viz.hpp"

#include <epoxy/gl_generated.h>
#include <pangolin/display/display.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/windowing/window.h>
#include <spdlog/spdlog.h>

using namespace wslam;

std::expected<Viz, std::string> Viz::initialize(VizInitOpts opts) {
    (void)opts;

    try {
        spdlog::info("[Viz] Creating new window");

        pangolin::CreateWindowAndBind(kWindowTitle);
        glEnable(GL_DEPTH_TEST);

        // Define Projection and initial ModelView matrix
        auto state_cam = std::make_unique<pangolin::OpenGlRenderState>(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

        // Create Interactive View in window
        auto handler = std::make_unique<pangolin::Handler3D>(*state_cam);

        pangolin::View* display_cam
            = &pangolin::CreateDisplay()
                   .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                   .SetHandler(handler.get());

        spdlog::debug("[Viz] Finished creating window");

        return Viz(std::move(state_cam), std::move(handler), display_cam);
    } catch (std::exception& e) {
        return std::unexpected(e.what());
    }
}

void Viz::addRequestNextCallback(std::function<void()> callback) {
    pangolin::RegisterKeyPressCallback('n', std::move(callback));
}

bool Viz::windowShouldClose() { return pangolin::ShouldQuit(); }

void Viz::closeWindow() { pangolin::Quit(); }

void Viz::startFrame() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    display_cam_->Activate(*state_cam_);
}

void Viz::drawTestCube() {
    spdlog::debug("[Viz] drawing a coloured cube");
    pangolin::glDrawColouredCube();
}

void Viz::drawTexture(const VizTexture& texture) {
    assert(texture.channels == 1
           || texture.channels == 3 && "Either 1 or 3 channels");

    const uint8_t* draw_data = texture.data.data();
    std::vector<uint8_t> expanded;

    // Expand grayscale to rgb
    if (texture.channels == 1) {
        expanded.reserve(3 * texture.data.size());
        for (uint8_t pix : texture.data) {
            expanded.push_back(pix);
            expanded.push_back(pix);
            expanded.push_back(pix);
        }
        draw_data = expanded.data();
    }

    // Upload to GPU
    pangolin::GlTexture gl_texture(static_cast<int>(texture.width),
                                   static_cast<int>(texture.height),
                                   GL_RGB8,  // internal format
                                   false,    // no mip maps
                                   0,        // border
                                   GL_RGB,   // source format
                                   GL_UNSIGNED_BYTE);

    gl_texture.Upload(draw_data, GL_RGB, GL_UNSIGNED_BYTE);

    // Render as a fullscreen quad in the current viewport
    display_cam_->Activate();
    gl_texture.RenderToViewportFlipY();
}

void Viz::endFrame() { pangolin::FinishFrame(); }
