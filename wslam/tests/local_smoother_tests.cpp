// Tests of the v2 mapping back-end machinery that need no GPU or dataset: the
// single-job worker thread and the sliding-window smoother

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

#include "job_worker.hpp"
#include "map_v2/map_common.hpp"
#include "map_v2/smoother_worker.hpp"

namespace {

using namespace wslam::v2::mapping;

// ---------------------------------------------------------------- JobWorker

TEST(JobWorkerTest, ReturnsTheJobResult) {
    wslam::base::JobWorker<int, int> worker(
        [](int&& value) { return value * 2; });

    EXPECT_EQ(worker.submit(21).get(), 42);
    // Consuming the future frees the worker for the next job
    EXPECT_EQ(worker.submit(5).get(), 10);
}

TEST(JobWorkerTest, StoresTheJobExceptionInTheFuture) {
    wslam::base::JobWorker<int, int> worker(
        [](int&&) -> int { throw std::runtime_error("job failed"); });

    auto future = worker.submit(1);
    EXPECT_THROW(future.get(), std::runtime_error);
    // The worker survives a throwing job
    EXPECT_THROW(worker.submit(2).get(), std::runtime_error);
}

// Blocks the worker on its first job, then submits a second one
void SubmitTwiceWhileBusy() {
    std::promise<void> release;
    const std::shared_future<void> released = release.get_future().share();
    wslam::base::JobWorker<int, int> worker([released](int&& value) {
        released.wait();
        return value;
    });
    auto first = worker.submit(1);
    auto second = worker.submit(2);
}

TEST(JobWorkerDeathTest, RejectsASecondJobInFlight) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(SubmitTwiceWhileBusy(), "previous job still in flight");
}

// ---------------------------------------------------------------- Smoother

const auto kPriorNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
const auto kOdometryNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2);

gtsam::Pose3 TruePose(size_t index) {
    return {gtsam::Rot3::Yaw(0.1 * static_cast<double>(index)),
            gtsam::Point3(static_cast<double>(index), 0.0, 0.0)};
}

// Bundle adding pose `index` (from 1), chained to the previous one by an
// exact odometry factor, x1 anchored by a prior. Initial values are perturbed
FactorBundle ChainBundle(size_t index) {
    const PoseId id{index};
    FactorBundle bundle;
    if (index == 1) {
        bundle.new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            Pose(id), TruePose(index), kPriorNoise);
    } else {
        const PoseId prev{index - 1};
        bundle.new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            Pose(prev), Pose(id), TruePose(index - 1).between(TruePose(index)),
            kOdometryNoise);
    }
    bundle.new_values.insert(
        Pose(id), TruePose(index).compose(gtsam::Pose3(
                      gtsam::Rot3::Roll(0.02), gtsam::Point3(0.05, 0.0, 0.0))));
    bundle.timestamps.emplace(Pose(id), SmootherTime(id));
    return bundle;
}

TEST(LocalSmootherTest, KeepsTheLastKeyframesOfTheWindow) {
    constexpr uint32_t kWindow = 3;
    SmootherWorker worker(SmootherWorker::Params{.window_keyframes = kWindow});

    SmootherWorker::Result result;
    for (size_t index = 1; index <= 6; ++index) {
        result
            = worker
                  .submit({.bundle = ChainBundle(index), .keyframe_id = index})
                  .get();
        ASSERT_FALSE(result.error.has_value()) << *result.error;
    }

    EXPECT_EQ(result.latest_values.size(), kWindow);
    for (size_t index = 4; index <= 6; ++index) {
        const gtsam::Key key = Pose(PoseId{index});
        ASSERT_TRUE(result.latest_values.exists(key))
            << "x" << index << " left the window early";
        // The prior on x1 survives in the marginal, so the chain stays exact
        EXPECT_TRUE(result.latest_values.at<gtsam::Pose3>(key).equals(
            TruePose(index), 1e-3))
            << "x" << index;
    }
}

TEST(LocalSmootherTest, FactorsTouchingMarginalisedKeysLeaveTheGraph) {
    SmootherWorker worker(SmootherWorker::Params{.window_keyframes = 2});

    FactorBundle first = ChainBundle(1);
    const gtsam::NonlinearFactor* prior = first.new_factors.at(0).get();

    auto result
        = worker.submit({.bundle = std::move(first), .keyframe_id = 1}).get();
    ASSERT_FALSE(result.error.has_value()) << *result.error;
    EXPECT_TRUE(result.live_slots.contains(prior));

    result = worker.submit({.bundle = ChainBundle(2), .keyframe_id = 2}).get();
    ASSERT_FALSE(result.error.has_value()) << *result.error;
    EXPECT_TRUE(result.live_slots.contains(prior));

    // x1 leaves a 2-keyframe window when x3 arrives; its prior goes with it
    result = worker.submit({.bundle = ChainBundle(3), .keyframe_id = 3}).get();
    ASSERT_FALSE(result.error.has_value()) << *result.error;
    EXPECT_FALSE(result.live_slots.contains(prior));
    EXPECT_FALSE(result.latest_values.exists(Pose(PoseId{1})));
}

TEST(LocalSmootherTest, RemoveAndReAddReusesSlots) {
    LocalSmoother smoother(/*smootherLag=*/100.0);

    FactorBundle first = ChainBundle(1);
    smoother.update(first.new_factors, first.new_values, first.timestamps,
                    std::set<size_t>{});
    FactorBundle second = ChainBundle(2);
    smoother.update(second.new_factors, second.new_values, second.timestamps,
                    std::set<size_t>{});
    const size_t slots = smoother.getFactors().size();

    // Replace the odometry factor x1-x2 over and over, like a re-observed
    // smart factor
    size_t odometry_slot = 1;
    for (int round = 0; round < 50; ++round) {
        gtsam::NonlinearFactorGraph replacement;
        replacement.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            Pose(PoseId{1}), Pose(PoseId{2}), TruePose(1).between(TruePose(2)),
            kOdometryNoise);
        const gtsam::NonlinearFactor* added = replacement.at(0).get();

        smoother.update(replacement, {}, {}, std::set<size_t>{odometry_slot});

        const auto& factors = smoother.getFactors();
        ASSERT_EQ(factors.size(), slots) << "round " << round;
        bool found = false;
        for (size_t slot = 0; slot < factors.size(); ++slot) {
            if (factors.at(slot).get() == added) {
                odometry_slot = slot;
                found = true;
            }
        }
        ASSERT_TRUE(found) << "round " << round;
    }
}

}  // namespace
