#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "data/provider_base.hpp"
#include "data/uzh_fpv_provider.hpp"

using namespace wslam;
namespace fs = std::filesystem;

namespace {

void WriteFile(const fs::path& path, const std::string& text) {
    std::ofstream file(path);
    file << text;
}

// A three-frame UZH-FPV sequence in the text format plus its Kalibr
// calibration, written to a fresh directory
class UzhFpvProviderTest : public ::testing::Test {
   protected:
    void SetUp() override {
        root_ = fs::path(::testing::TempDir())
                / ("uzh_fpv_"
                   + std::string(::testing::UnitTest::GetInstance()
                                     ->current_test_info()
                                     ->name()));
        fs::remove_all(root_);
        fs::create_directories(sequence() / "img");
        fs::create_directories(calibration());

        // As in the downloads: the IMU header leaves out the id column, the
        // image lists end their lines in a space
        WriteFile(sequence() / "imu.txt",
                  "# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x "
                  "lin_acc_y lin_acc_z\n"
                  "0 1398.500000000000 0.1 0.2 0.3 1.0 2.0 9.8\n"
                  "1 1398.510000000000 0.1 0.2 0.3 1.0 2.0 9.8\n"
                  "2 1398.540000000000 0.1 0.2 0.3 1.0 2.0 9.8\n");
        for (const auto& [list, camera] : {std::pair{"left_images.txt", 0},
                                           std::pair{"right_images.txt", 1}}) {
            WriteFile(sequence() / list,
                      std::format("# id timestamp image_name\n"
                                  "0 1398.520000000000 img/image_{0}_0.png \n"
                                  "1 1398.553333333000 img/image_{0}_1.png \n",
                                  camera));
        }

        const std::string camera_block = R"(
  T_cam_imu:
  - [0.0, -1.0, 0.0, 0.1]
  - [1.0, 0.0, 0.0, 0.2]
  - [0.0, 0.0, 1.0, 0.3]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_coeffs: [0.01, -0.02, 0.03, -0.04]
  distortion_model: equidistant
  intrinsics: [275.0, 274.0, 316.0, 242.0]
  resolution: [640, 480]
  timeshift_cam_imu: -0.015
)";
        WriteFile(calibration() / "camchain-imucam-..calib_imu.yaml",
                  "cam0:" + camera_block + "cam1:" + camera_block);
        // As in the downloads: the imu0 block without its key
        WriteFile(calibration() / "imu.yaml", R"(  T_i_b:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 0.1
  accelerometer_random_walk: 0.002
  gyroscope_noise_density: 0.05
  gyroscope_random_walk: 4.0e-05
  update_rate: 500.0
)");
    }

    void TearDown() override { fs::remove_all(root_); }

    [[nodiscard]] fs::path sequence() const { return root_ / "sequence"; }
    [[nodiscard]] fs::path calibration() const { return root_ / "calib"; }

    [[nodiscard]] data::UzhFpvProvider makeProvider() const {
        return data::UzhFpvProvider(data::UzhFpvProviderOpts{
            .sequence_dir = sequence(), .calibration_dir = calibration()});
    }

   private:
    fs::path root_;
};

}  // namespace

TEST(UzhFpvTimestampTest, ParsesSecondsExactly) {
    EXPECT_EQ(data::ParseSecondsToNanoseconds("1398.526170378000"),
              1398526170378ULL);
    EXPECT_EQ(data::ParseSecondsToNanoseconds("1545305703.6524107"),
              1545305703652410700ULL);
    EXPECT_EQ(data::ParseSecondsToNanoseconds("12"), 12000000000ULL);
    // Past the nanosecond digits are truncated
    EXPECT_EQ(data::ParseSecondsToNanoseconds("1.0000000019"), 1000000001ULL);
    EXPECT_FALSE(data::ParseSecondsToNanoseconds("-1.5").has_value());
    EXPECT_FALSE(data::ParseSecondsToNanoseconds("1e9").has_value());
    EXPECT_FALSE(data::ParseSecondsToNanoseconds("").has_value());
}

TEST_F(UzhFpvProviderTest, ReadsTheKalibrCalibration) {
    auto provider = makeProvider();
    const auto params = provider.getSensorParams();
    ASSERT_TRUE(params.has_value()) << params.error();
    ASSERT_EQ(params->cams.size(), 2U);

    const data::CamSensorParams& camera = params->cams[0];
    EXPECT_EQ(camera.distortion_model, data::DistortionModel::Equidistant);
    EXPECT_EQ(camera.resolution, (std::array<int, 2>{640, 480}));
    EXPECT_EQ(camera.intrinsics, Eigen::Vector4d(275.0, 274.0, 316.0, 242.0));
    EXPECT_EQ(camera.distortion_coefficients,
              Eigen::Vector4d(0.01, -0.02, 0.03, -0.04));
    EXPECT_EQ(camera.rate_hz, 30U);

    // T_BS is camera -> IMU (= body): the inverse of Kalibr's T_cam_imu
    Eigen::Matrix4d cam_from_imu;
    cam_from_imu << 0, -1, 0, 0.1, 1, 0, 0, 0.2, 0, 0, 1, 0.3, 0, 0, 0, 1;
    EXPECT_TRUE((camera.T_BS * cam_from_imu).isIdentity(1e-12));

    EXPECT_EQ(params->imu.rate_hz, 500U);
    EXPECT_DOUBLE_EQ(params->imu.gyroscope_noise_density, 0.05);
    EXPECT_DOUBLE_EQ(params->imu.accelerometer_random_walk, 0.002);
    EXPECT_TRUE(params->imu.T_BS.isIdentity());
}

TEST_F(UzhFpvProviderTest, InterleavesImuAndShiftedFramesByTime) {
    auto provider = makeProvider();

    struct Item {
        bool is_imu;
        uint64_t timestamp;
    };
    std::vector<Item> items;
    for (auto reading : provider.getReadings()) {
        ASSERT_TRUE(reading.has_value()) << reading.error();
        if (reading->imu) {
            items.push_back(
                {.is_imu = true, .timestamp = reading->imu->timestamp});
            EXPECT_FLOAT_EQ(reading->imu->wz(), 0.3F);
            EXPECT_FLOAT_EQ(reading->imu->az(), 9.8F);
        }
        if (reading->frames[0]) {
            // The stereo pair stays one reading under the shared shift
            ASSERT_TRUE(reading->frames[1].has_value());
            EXPECT_EQ(reading->frames[0]->timestamp,
                      reading->frames[1]->timestamp);
            items.push_back(
                {.is_imu = false, .timestamp = reading->frames[0]->timestamp});
        }
    }

    // Frames at 1398.520 and 1398.553333333 s move 15 ms earlier onto the IMU
    // clock: 1398.505 lands between the first two IMU samples
    const std::vector<uint64_t> expected{1398500000000ULL, 1398505000000ULL,
                                         1398510000000ULL, 1398538333333ULL,
                                         1398540000000ULL};
    ASSERT_EQ(items.size(), expected.size());
    for (size_t i = 0; i < items.size(); ++i) {
        EXPECT_EQ(items[i].timestamp, expected[i]) << "item " << i;
    }
    EXPECT_FALSE(items[1].is_imu);
    EXPECT_FALSE(items[3].is_imu);
}

TEST_F(UzhFpvProviderTest, ReportsAMissingCalibration) {
    fs::remove(calibration() / "imu.yaml");
    auto provider = makeProvider();
    const auto params = provider.getSensorParams();
    ASSERT_FALSE(params.has_value());
    EXPECT_NE(params.error().find("imu.yaml"), std::string::npos)
        << params.error();
}
