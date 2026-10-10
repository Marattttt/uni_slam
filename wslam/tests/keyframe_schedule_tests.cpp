#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "data/provider_base.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"
#include "keyframe_schedule.hpp"

using namespace wslam;
using Kind = data::ReadingCandidate::Kind;
using Decision = data::ReadingDecision;

namespace {

constexpr const char* kV101Schedule
    = WSLAM_TEST_DATA_DIR "/v1_keyframes_V101.json";

// Writes `text` to a file in a per-test temporary directory
class ScheduleFileTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info
            = ::testing::UnitTest::GetInstance()->current_test_info();
        directory_ = std::filesystem::temp_directory_path()
                     / (std::string("wslam_keyframe_schedule_") + info->name());
        std::filesystem::create_directories(directory_);
    }

    void TearDown() override { std::filesystem::remove_all(directory_); }

    std::filesystem::path write(std::string_view text) {
        const auto path = directory_ / "schedule.json";
        std::ofstream(path) << text;
        return path;
    }

    void expectError(std::string_view text, std::string_view fragment) {
        const auto schedule = LoadKeyframeSchedule(write(text));
        ASSERT_FALSE(schedule.has_value()) << text;
        EXPECT_TRUE(schedule.error().contains(fragment))
            << "'" << schedule.error() << "' should mention '" << fragment
            << "'";
    }

   private:
    std::filesystem::path directory_;
};

data::ReadingCandidate Frame(uint64_t timestamp, uint64_t index) {
    return {.kind = Kind::Frame, .timestamp = timestamp, .index = index};
}

data::ReadingCandidate Imu(uint64_t timestamp, uint64_t index) {
    return {.kind = Kind::Imu, .timestamp = timestamp, .index = index};
}

KeyframeSchedule TwoKeyframes() {
    return KeyframeSchedule{
        .sequence = "test",
        .frame_indices = {2, 5},
        .timestamps_ns = {200, 500},
    };
}

}  // namespace

TEST(KeyframeScheduleTest, LoadsTheV101Schedule) {
    const auto schedule = LoadKeyframeSchedule(kV101Schedule);
    ASSERT_TRUE(schedule.has_value()) << schedule.error();

    EXPECT_EQ(schedule->sequence, "V1_01_easy");
    ASSERT_EQ(schedule->frame_indices.size(), 50U);
    ASSERT_EQ(schedule->timestamps_ns.size(), 50U);
    EXPECT_EQ(schedule->frame_indices.front(), 670U);
    EXPECT_EQ(schedule->timestamps_ns.front(), 1403715306762142976U);
    EXPECT_EQ(schedule->frame_indices.back(), 830U);
    EXPECT_EQ(schedule->timestamps_ns.back(), 1403715314762142976U);
}

TEST_F(ScheduleFileTest, LoadsAWellFormedFile) {
    const auto schedule = LoadKeyframeSchedule(
        write(R"({"seq": {"keyframes": [1, 4], "timestamps_ns": [10, 40]}})"));
    ASSERT_TRUE(schedule.has_value()) << schedule.error();
    EXPECT_EQ(schedule->sequence, "seq");
    EXPECT_EQ(schedule->frame_indices, (std::vector<uint64_t>{1, 4}));
    EXPECT_EQ(schedule->timestamps_ns, (std::vector<uint64_t>{10, 40}));
}

TEST_F(ScheduleFileTest, RejectsMalformedFiles) {
    expectError("{not json", "parsing");
    expectError("[]", "exactly one sequence");
    expectError(R"({"a": {"keyframes": [1], "timestamps_ns": [1]},
                    "b": {"keyframes": [1], "timestamps_ns": [1]}})",
                "exactly one sequence");
    expectError(R"({"a": 3})", "not an object");
    expectError(R"({"a": {"timestamps_ns": [1]}})", "'keyframes'");
    expectError(R"({"a": {"keyframes": [], "timestamps_ns": []}})", "empty");
    expectError(R"({"a": {"keyframes": [1, -2], "timestamps_ns": [1, 2]}})",
                "unsigned");
    expectError(R"({"a": {"keyframes": [1, 1.5], "timestamps_ns": [1, 2]}})",
                "unsigned");
    expectError(R"({"a": {"keyframes": [2, 1], "timestamps_ns": [1, 2]}})",
                "strictly increasing");
    expectError(R"({"a": {"keyframes": [1, 2], "timestamps_ns": [5, 5]}})",
                "strictly increasing");
    expectError(R"({"a": {"keyframes": [1, 2], "timestamps_ns": [5]}})",
                "2 keyframes but 1 timestamps");
}

TEST(KeyframeScheduleTest, RejectsAMissingFile) {
    const auto schedule
        = LoadKeyframeSchedule("/nonexistent/wslam/keyframes.json");
    ASSERT_FALSE(schedule.has_value());
    EXPECT_TRUE(schedule.error().contains("could not open"));
}

TEST(KeyframeFilterTest, KeepsScheduledFramesAndAllImu) {
    auto filter = MakeKeyframeFilter(TwoKeyframes());

    EXPECT_EQ(filter(Imu(50, 0)), Decision::Keep);
    EXPECT_EQ(filter(Frame(100, 1)), Decision::Skip);
    EXPECT_EQ(filter(Imu(150, 1)), Decision::Keep);
    EXPECT_EQ(filter(Frame(200, 2)), Decision::Keep);
    EXPECT_EQ(filter(Frame(300, 3)), Decision::Skip);
    EXPECT_EQ(filter(Frame(400, 4)), Decision::Skip);
    EXPECT_EQ(filter(Frame(500, 5)), Decision::Keep);
    EXPECT_EQ(filter(Imu(550, 9)), Decision::Keep);
    // The first frame past the last keyframe ends the stream
    EXPECT_EQ(filter(Frame(600, 6)), Decision::StopStream);
    EXPECT_EQ(filter(Frame(700, 7)), Decision::StopStream);
}

TEST(KeyframeFilterTest, RejectsAKeyframeAtTheWrongIndex) {
    auto filter = MakeKeyframeFilter(TwoKeyframes());

    const auto decision = filter(Frame(200, 3));
    ASSERT_FALSE(decision.has_value());
    EXPECT_TRUE(decision.error().contains("schedule says frame 2"))
        << decision.error();
}

TEST(KeyframeFilterTest, RejectsAKeyframeMissingFromTheStream) {
    auto filter = MakeKeyframeFilter(TwoKeyframes());

    EXPECT_EQ(filter(Frame(100, 1)), Decision::Skip);
    const auto decision = filter(Frame(250, 2));
    ASSERT_FALSE(decision.has_value());
    EXPECT_TRUE(decision.error().contains("not in the stream"))
        << decision.error();
}

TEST(RacoLightglueTest, KeypointsMapToSourcePixels) {
    namespace model = wslam::v2::raco_lightglue;

    // The centre of the model input's top-left pixel is the centre of the
    // source pixel at the crop offset
    const Eigen::Vector2d top_left
        = model::ToSourcePixel(model::KeypointPosition{.x = 0.5F, .y = 0.5F});
    EXPECT_DOUBLE_EQ(top_left.x(), static_cast<double>(model::kCropX));
    EXPECT_DOUBLE_EQ(top_left.y(), static_cast<double>(model::kCropY));

    // 752 x 480 cropped to 736 x 480
    EXPECT_EQ(model::kCropX, 8U);
    EXPECT_EQ(model::kCropY, 0U);

    const Eigen::Vector2d bottom_right = model::ToSourcePixel(
        model::KeypointPosition{.x = static_cast<float>(model::kInputWidth),
                                .y = static_cast<float>(model::kInputHeight)});
    EXPECT_DOUBLE_EQ(bottom_right.x(),
                     model::kCropX + model::kInputWidth - 0.5);
    EXPECT_DOUBLE_EQ(bottom_right.y(),
                     model::kCropY + model::kInputHeight - 0.5);
}
