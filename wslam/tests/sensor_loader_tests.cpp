#include <gtest/gtest.h>

#include <cstdint>
#include <expected>
#include <generator>
#include <memory>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "common.hpp"
#include "compute/compute.hpp"
#include "data/provider_base.hpp"
#include "feature_detect_v2/sensor_loader.hpp"

using namespace wslam;
using Kind = data::ReadingCandidate::Kind;
using Decision = data::ReadingDecision;
using ReadingType = std::expected<data::Reading<1>, std::string>;

namespace {

// One stream item: an IMU sample or a frame, at a timestamp
struct Item {
    Kind kind;
    uint64_t timestamp;
};

// Readings for `items`. A frame's pixels are "read" only when its load() is
// called, which appends its timestamp to `loaded`
std::generator<ReadingType> Stream(
    std::vector<Item> items, std::shared_ptr<std::vector<uint64_t>> loaded) {
    for (const Item& item : items) {
        data::Reading<1> reading;
        if (item.kind == Kind::Imu) {
            reading.imu
                = data::IMUReading{.timestamp = item.timestamp, .vals = {}};
        } else {
            reading.frames[0] = data::PendingFrame{
                .timestamp = item.timestamp,
                .load = [loaded, timestamp = item.timestamp]()
                    -> std::expected<data::FrameBW, std::string> {
                    loaded->push_back(timestamp);
                    return data::FrameBW{
                        .timestamp = timestamp,
                        .pixels = {},
                        .width = 0,
                        .height = 0,
                    };
                },
            };
        }
        co_yield reading;
    }
}

// IMU at 10, 20, ... and a frame every third item: frames at 30, 60, 90
std::vector<Item> ImuAndFrames() {
    std::vector<Item> items;
    for (uint64_t timestamp = 10; timestamp <= 90; timestamp += 10) {
        items.push_back(Item{
            .kind = timestamp % 30 == 0 ? Kind::Frame : Kind::Imu,
            .timestamp = timestamp,
        });
    }
    return items;
}

class SensorLoaderTest : public ::testing::Test {
   protected:
    std::shared_ptr<std::vector<uint64_t>> loaded_
        = std::make_shared<std::vector<uint64_t>>();
    AnyBag storage_;

    std::unique_ptr<v2::SensorLoaderPass> makeLoader(std::vector<Item> items,
                                                     data::ReadingFilter filter
                                                     = {}) {
        auto loader = std::make_unique<v2::SensorLoaderPass>(
            Stream(std::move(items), loaded_), storage_, std::move(filter));
        EXPECT_EQ(loader->initialize(), std::nullopt);
        return loader;
    }

    [[nodiscard]] uint64_t frameTimestamp() const {
        return storage_.get<data::FrameBW>(ResourceIdentifier::GetFrameName(0))
            .value()
            .timestamp;
    }

    [[nodiscard]] std::vector<uint64_t> imuTimestamps() const {
        std::vector<uint64_t> timestamps;
        for (const auto& imu : storage_
                                   .get<std::vector<data::IMUReading>>(
                                       ResourceIdentifier::GetImuVecName())
                                   .value()) {
            timestamps.push_back(imu.timestamp);
        }
        return timestamps;
    }
};

}  // namespace

TEST_F(SensorLoaderTest, WithoutFilterKeepsEverything) {
    auto loader = makeLoader(ImuAndFrames());

    for (const uint64_t frame : {30U, 60U, 90U}) {
        ASSERT_EQ(loader->execute(), std::nullopt);
        EXPECT_EQ(frameTimestamp(), frame);
        EXPECT_EQ(
            storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName),
            frame);
    }
    EXPECT_EQ(imuTimestamps(), (std::vector<uint64_t>{10, 20, 40, 50, 70, 80}));
    EXPECT_EQ(*loaded_, (std::vector<uint64_t>{30, 60, 90}));

    // The end of the stream stops the pipeline, every time it is asked
    EXPECT_EQ(loader->execute(), compute::kComputeStopExecution);
    EXPECT_EQ(loader->execute(), compute::kComputeStopExecution);
}

TEST_F(SensorLoaderTest, SkippedFramesAreNeverRead) {
    auto filter = [](const data::ReadingCandidate& candidate)
        -> std::expected<Decision, std::string> {
        if (candidate.kind == Kind::Frame && candidate.timestamp == 60) {
            return Decision::Skip;
        }
        return Decision::Keep;
    };
    auto loader = makeLoader(ImuAndFrames(), filter);

    ASSERT_EQ(loader->execute(), std::nullopt);
    EXPECT_EQ(frameTimestamp(), 30U);

    // The IMU around the skipped frame arrives with the next kept one
    ASSERT_EQ(loader->execute(), std::nullopt);
    EXPECT_EQ(frameTimestamp(), 90U);
    EXPECT_EQ(imuTimestamps(), (std::vector<uint64_t>{10, 20, 40, 50, 70, 80}));

    EXPECT_EQ(*loaded_, (std::vector<uint64_t>{30, 90}));
    EXPECT_EQ(loader->execute(), compute::kComputeStopExecution);
}

TEST_F(SensorLoaderTest, LeadInSkipDropsImuAndFrames) {
    // Stands in for "skip the first N seconds" of a dataset
    constexpr uint64_t kStart = 45;
    auto filter = [](const data::ReadingCandidate& candidate)
        -> std::expected<Decision, std::string> {
        return candidate.timestamp < kStart ? Decision::Skip : Decision::Keep;
    };
    auto loader = makeLoader(ImuAndFrames(), filter);

    ASSERT_EQ(loader->execute(), std::nullopt);
    EXPECT_EQ(frameTimestamp(), 60U);
    EXPECT_EQ(imuTimestamps(), (std::vector<uint64_t>{50}));
    EXPECT_EQ(*loaded_, (std::vector<uint64_t>{60}));
}

TEST_F(SensorLoaderTest, FilterSeesPerKindIndices) {
    auto seen = std::make_shared<std::vector<data::ReadingCandidate>>();
    auto filter = [seen](const data::ReadingCandidate& candidate)
        -> std::expected<Decision, std::string> {
        seen->push_back(candidate);
        return candidate.kind == Kind::Frame && candidate.index == 1
                   ? Decision::Skip
                   : Decision::Keep;
    };
    auto loader = makeLoader(ImuAndFrames(), filter);

    while (loader->execute() == std::nullopt) {
    }

    std::vector<uint64_t> frame_indices;
    std::vector<uint64_t> imu_indices;
    for (const auto& candidate : *seen) {
        (candidate.kind == Kind::Frame ? frame_indices : imu_indices)
            .push_back(candidate.index);
    }
    // Skipped items still count
    EXPECT_EQ(frame_indices, (std::vector<uint64_t>{0, 1, 2}));
    EXPECT_EQ(imu_indices, (std::vector<uint64_t>{0, 1, 2, 3, 4, 5}));
}

TEST_F(SensorLoaderTest, StopStreamEndsThePipeline) {
    auto filter = [](const data::ReadingCandidate& candidate)
        -> std::expected<Decision, std::string> {
        return candidate.kind == Kind::Frame && candidate.timestamp > 30
                   ? Decision::StopStream
                   : Decision::Keep;
    };
    auto loader = makeLoader(ImuAndFrames(), filter);

    ASSERT_EQ(loader->execute(), std::nullopt);
    EXPECT_EQ(frameTimestamp(), 30U);

    EXPECT_EQ(loader->execute(), compute::kComputeStopExecution);
    EXPECT_EQ(loader->execute(), compute::kComputeStopExecution);
    EXPECT_EQ(*loaded_, (std::vector<uint64_t>{30}));
}

TEST_F(SensorLoaderTest, FilterErrorsPropagate) {
    auto filter = [](const data::ReadingCandidate& candidate)
        -> std::expected<Decision, std::string> {
        if (candidate.kind == Kind::Frame) {
            return std::unexpected("bad frame");
        }
        return Decision::Keep;
    };
    auto loader = makeLoader(ImuAndFrames(), filter);

    const auto err = loader->execute();
    ASSERT_TRUE(err.has_value());
    EXPECT_NE(err.value(), compute::kComputeStopExecution);
    EXPECT_TRUE(err->contains("bad frame")) << err.value();
    EXPECT_TRUE(loaded_->empty());
}

TEST_F(SensorLoaderTest, StreamWithoutFramesStops) {
    auto loader = makeLoader({Item{.kind = Kind::Imu, .timestamp = 10}});

    EXPECT_EQ(loader->execute(), compute::kComputeStopExecution);
}
