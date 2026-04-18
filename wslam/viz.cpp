#include "viz.hpp"

#include <epoxy/gl_generated.h>
#include <pangolin/display/display.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/windowing/window.h>
#include <spdlog/spdlog.h>

#include <cmath>

using namespace wslam;
using namespace wslam::viz;

#define LOG_ID "[VizGUI]"

std::expected<VizGUI, std::string> VizGUI::create(VizInitOpts opts) {
    (void)opts;

    try {
        spdlog::info(LOG_ID " Creating new window");

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

        spdlog::debug(LOG_ID " Finished creating window");

        return VizGUI(std::move(state_cam), std::move(handler), display_cam);
    } catch (std::exception& e) {
        return std::unexpected(e.what());
    }
}

void VizGUI::addRequestNextCallback(std::function<void()> callback) {
    pangolin::RegisterKeyPressCallback('n', std::move(callback));
}

bool VizGUI::windowShouldClose() { return pangolin::ShouldQuit(); }

void VizGUI::closeWindow() {
    spdlog::info(LOG_ID " Closing window");
    pangolin::Quit();
}

void VizGUI::startFrame() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    display_cam_->Activate(*state_cam_);
}

void VizGUI::drawTestCube() {
    spdlog::debug(LOG_ID " drawing a coloured cube");
    pangolin::glDrawColouredCube();
}

void VizGUI::drawTexture(const VizTexture& texture) {
    const data::PixelRGB* draw_data = texture.data.data();

    static_assert(std::is_same_v<decltype(data::PixelRGB::r), uint8_t>
                  && std::is_same_v<decltype(data::PixelRGB::g), uint8_t>
                  && std::is_same_v<decltype(data::PixelRGB::b), uint8_t>);

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

void VizGUI::endFrame() { pangolin::FinishFrame(); }

void VizGUI::drawFeatures(Features features) {
    if (features.features.empty() || features.image_width == 0
        || features.image_height == 0) {
        return;
    }

    display_cam_->Activate();

    // Switch to 2D image-pixel coordinates. Flip Y so (0,0) is top-left
    // to match the way drawTexture + RenderToViewportFlipY presents the image.
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(features.image_width),
            static_cast<double>(features.image_height), 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    const bool depth_was_enabled = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

    glColor3ub(features.color[0], features.color[1], features.color[2]);
    constexpr float kLineWidth = 1.5F;
    glLineWidth(kLineWidth);

    constexpr int kSegments = 16;
    constexpr float kTwoPi = 6.28318530718F;
    for (const auto& feat : features.features) {
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < kSegments; ++i) {
            const float theta = kTwoPi * static_cast<float>(i) / kSegments;
            glVertex2f(feat.x + features.radius * std::cos(theta),
                       feat.y + features.radius * std::sin(theta));
        }
        glEnd();
    }

    if (depth_was_enabled) {
        glEnable(GL_DEPTH_TEST);
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}
