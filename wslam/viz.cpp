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

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    gl_texture.Upload(draw_data, GL_RGB, GL_UNSIGNED_BYTE);

    // Render as a fullscreen quad in the current viewport
    display_cam_->Activate();
    gl_texture.RenderToViewportFlipY();
}

void VizGUI::endFrame() { pangolin::FinishFrame(); }

void VizGUI::drawCorners(Corners corners) {
    if (corners.corners.empty() || corners.image_width == 0
        || corners.image_height == 0) {
        return;
    }

    display_cam_->Activate();

    // Switch to 2D image-pixel coordinates. Flip Y so (0,0) is top-left
    // to match the way drawTexture + RenderToViewportFlipY presents the image.
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(corners.image_width),
            static_cast<double>(corners.image_height), 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    const bool depth_was_enabled = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

    glColor3ub(corners.color[0], corners.color[1], corners.color[2]);
    constexpr float kLineWidth = 1.5F;
    glLineWidth(kLineWidth);

    constexpr int kSegments = 16;
    constexpr float kTwoPi = 6.28318530718F;
    for (const auto& feat : corners.corners) {
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < kSegments; ++i) {
            const float theta = kTwoPi * static_cast<float>(i) / kSegments;
            glVertex2f(feat.x + corners.radius * std::cos(theta),
                       feat.y + corners.radius * std::sin(theta));
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

void VizGUI::drawFeatures(VizFeatures features) {
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

    constexpr int kSegments = 16;
    constexpr float kTwoPi = 6.28318530718F;
    constexpr size_t kDescriptorBits = 256;
    constexpr float kDescriptorLineWidth = 1.0F;
    constexpr float kFeatureLineWidth = 1.5F;

    for (const auto& feat : features.features) {
        // NOTE: Feature uses {width, height} as pixel coords here.
        const float fx = static_cast<float>(feat.x);
        const float fy = static_cast<float>(feat.y);
        const float cos_o = std::cos(feat.orientation);
        const float sin_o = std::sin(feat.orientation);

        // 1) BRIEF pattern: one line per bit, colored by bit value.
        //    Offsets in the test set are relative to the feature centre
        //    and rotated by the feature's orientation.
        glLineWidth(kDescriptorLineWidth);
        glBegin(GL_LINES);
        for (size_t bit = 0; bit < kDescriptorBits; ++bit) {
            const float x1 = static_cast<float>(features.test_set[bit * 4 + 0]);
            const float y1 = static_cast<float>(features.test_set[bit * 4 + 1]);
            const float x2 = static_cast<float>(features.test_set[bit * 4 + 2]);
            const float y2 = static_cast<float>(features.test_set[bit * 4 + 3]);

            const float rx1 = cos_o * x1 - sin_o * y1;
            const float ry1 = sin_o * x1 + cos_o * y1;
            const float rx2 = cos_o * x2 - sin_o * y2;
            const float ry2 = sin_o * x2 + cos_o * y2;

            const auto bits_in_uint32 = 32;
            const auto& c = (feat.descriptor[bit / bits_in_uint32]
                                 >> (bit % bits_in_uint32)
                             != 0U)
                                ? features.bit_one_color
                                : features.bit_zero_color;
            glColor3ub(c[0], c[1], c[2]);
            glVertex2f(fx + rx1, fy + ry1);
            glVertex2f(fx + rx2, fy + ry2);
        }
        glEnd();

        // 2) Feature circle + orientation tick on top.
        glColor3ub(features.feature_color[0], features.feature_color[1],
                   features.feature_color[2]);
        glLineWidth(kFeatureLineWidth);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < kSegments; ++i) {
            const float theta = kTwoPi * static_cast<float>(i) / kSegments;
            glVertex2f(fx + features.radius * std::cos(theta),
                       fy + features.radius * std::sin(theta));
        }
        glEnd();

        glBegin(GL_LINES);
        glVertex2f(fx, fy);
        glVertex2f(fx + features.radius * cos_o, fy + features.radius * sin_o);
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
