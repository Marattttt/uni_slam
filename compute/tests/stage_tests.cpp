#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "compute/compute.hpp"
#include "compute/gpu_stage.hpp"
#include "compute/pass.hpp"
#include "compute/stage.hpp"

using namespace wslam::compute;

namespace {

// Records lifecycle calls into caller-owned flags.
struct PassFlags {
    bool did_initialize = false;
    bool did_execute = false;
};

class FlagPass : public Pass {
   public:
    FlagPass(std::string id, PassFlags& flags)
        : id_(std::move(id)), flags_(&flags) {}

    [[nodiscard]] std::string getId() const override { return id_; }
    std::optional<std::string> initialize() override {
        flags_->did_initialize = true;
        return std::nullopt;
    }
    std::optional<std::string> execute() override {
        flags_->did_execute = true;
        return std::nullopt;
    }

   private:
    std::string id_;
    PassFlags* flags_;
};

// Appends its id to a shared order log on execute.
class OrderPass : public Pass {
   public:
    OrderPass(std::string id, std::vector<std::string>& order)
        : id_(std::move(id)), order_(&order) {}

    [[nodiscard]] std::string getId() const override { return id_; }
    std::optional<std::string> initialize() override { return std::nullopt; }
    std::optional<std::string> execute() override {
        order_->push_back(id_);
        return std::nullopt;
    }

   private:
    std::string id_;
    std::vector<std::string>* order_;
};

constexpr std::string kFailureMsg = "boom";

// Returns fixed values from initialize()/execute() (sentinels or errors).
class ReturningPass : public Pass {
   public:
    ReturningPass(std::string id, std::optional<std::string> on_execute,
                  std::optional<std::string> on_initialize = std::nullopt)
        : id_(std::move(id)),
          on_execute_(std::move(on_execute)),
          on_initialize_(std::move(on_initialize)) {}

    [[nodiscard]] std::string getId() const override { return id_; }
    std::optional<std::string> initialize() override { return on_initialize_; }
    std::optional<std::string> execute() override { return on_execute_; }

   private:
    std::string id_;
    std::optional<std::string> on_execute_;
    std::optional<std::string> on_initialize_;
};

// Writes a fixed value into an AnyBag under a key.
class WriteBagPass : public Pass {
   public:
    WriteBagPass(AnyBag& bag, std::string key, int value)
        : bag_(&bag), key_(std::move(key)), value_(value) {}

    [[nodiscard]] std::string getId() const override { return "write"; }
    std::optional<std::string> initialize() override { return std::nullopt; }
    std::optional<std::string> execute() override {
        bag_->set(key_, value_);
        return std::nullopt;
    }

   private:
    AnyBag* bag_;
    std::string key_;
    int value_;
};

// Reads a value from an AnyBag into a caller-owned slot.
class ReadBagPass : public Pass {
   public:
    ReadBagPass(AnyBag& bag, std::string key, std::optional<int>& out)
        : bag_(&bag), key_(std::move(key)), out_(&out) {}

    [[nodiscard]] std::string getId() const override { return "read"; }
    std::optional<std::string> initialize() override { return std::nullopt; }
    std::optional<std::string> execute() override {
        *out_ = bag_->get<int>(key_);
        return std::nullopt;
    }

   private:
    AnyBag* bag_;
    std::string key_;
    std::optional<int>* out_;
};

// An uninitialised GPU: cheap to construct (no Dawn device) as long as
// initialize() is never called. Suffices for GpuStage tests whose GPU batch is
// empty.
std::shared_ptr<GPU> makeUninitialisedGpu() {
    return std::make_shared<GPU>(nullptr, nullptr, ".");
}

}  // namespace

TEST(StageTest, ConstructsWithoutPerf) {
    Stage stage{"empty"};
    EXPECT_EQ(stage.getId(), "empty");
    EXPECT_EQ(stage.initialize(), std::nullopt);
    EXPECT_EQ(stage.execute(), std::nullopt);
}

TEST(StageTest, IdIsStoredVerbatimWithoutBrackets) {
    Stage stage{"Feature detect"};
    EXPECT_EQ(stage.getId(), "Feature detect");
}

TEST(StageTest, InitializeAndExecuteRunEveryPass) {
    PassFlags a;
    PassFlags b;
    Stage stage{"s"};
    stage.add_pass(std::make_unique<FlagPass>("a", a));
    stage.add_pass(std::make_unique<FlagPass>("b", b));

    ASSERT_EQ(stage.initialize(), std::nullopt);
    EXPECT_TRUE(a.did_initialize);
    EXPECT_TRUE(b.did_initialize);
    EXPECT_FALSE(a.did_execute);

    ASSERT_EQ(stage.execute(), std::nullopt);
    EXPECT_TRUE(a.did_execute);
    EXPECT_TRUE(b.did_execute);
}

TEST(StageTest, ExecuteRunsPassesInOrder) {
    std::vector<std::string> order;
    Stage stage{"s"};
    stage.add_pass(std::make_unique<OrderPass>("first", order));
    stage.add_pass(std::make_unique<OrderPass>("second", order));
    stage.add_pass(std::make_unique<OrderPass>("third", order));

    ASSERT_EQ(stage.execute(), std::nullopt);
    EXPECT_EQ(order, (std::vector<std::string>{"first", "second", "third"}));
}

TEST(StageTest, InitializeFailureNamesThePass) {
    Stage stage{"s"};
    stage.add_pass(
        std::make_unique<ReturningPass>("bad", std::nullopt, kFailureMsg));

    const auto err = stage.initialize();
    ASSERT_TRUE(err.has_value());
    EXPECT_NE(err->find("bad"), std::string::npos);
    EXPECT_NE(err->find(kFailureMsg), std::string::npos);
}

TEST(StageTest, ExecuteFailureNamesThePass) {
    Stage stage{"s"};
    stage.add_pass(std::make_unique<ReturningPass>("bad", kFailureMsg));

    const auto err = stage.execute();
    ASSERT_TRUE(err.has_value());
    EXPECT_NE(err->find("bad"), std::string::npos);
    EXPECT_NE(err->find(kFailureMsg), std::string::npos);
}

TEST(StageTest, StageStopHaltsRemainingPasses) {
    std::vector<std::string> order;
    Stage stage{"s"};
    stage.add_pass(std::make_unique<OrderPass>("before", order));
    stage.add_pass(
        std::make_unique<ReturningPass>("stopper", kStageStopExecution));
    stage.add_pass(std::make_unique<OrderPass>("after", order));

    // Stage-stop is not an error: execute() returns nullopt...
    EXPECT_EQ(stage.execute(), std::nullopt);
    // ...and the pass after the stopper never runs.
    EXPECT_EQ(order, (std::vector<std::string>{"before"}));
}

TEST(StageTest, ComputeStopPropagates) {
    Stage stage{"s"};
    stage.add_pass(
        std::make_unique<ReturningPass>("halt", kComputeStopExecution));

    const auto err = stage.execute();
    ASSERT_TRUE(err.has_value());
    EXPECT_EQ(err.value(), kComputeStopExecution);
}

TEST(StageTest, CustomPassRunsCallbackAndForwardsError) {
    bool ran = false;
    Stage stage{"s"};
    stage.add_pass(std::make_unique<CustomPass>(
        "custom", [&ran](CustomPass*) -> std::optional<std::string> {
            ran = true;
            return kFailureMsg;
        }));

    const auto err = stage.execute();
    EXPECT_TRUE(ran);
    ASSERT_TRUE(err.has_value());
    EXPECT_NE(err->find(kFailureMsg), std::string::npos);
}

TEST(StageTest, PassesShareDataThroughAnyBag) {
    AnyBag bag;
    std::optional<int> seen;
    Stage stage{"s"};
    stage.add_pass(std::make_unique<WriteBagPass>(bag, "k", 42));
    stage.add_pass(std::make_unique<ReadBagPass>(bag, "k", seen));

    ASSERT_EQ(stage.execute(), std::nullopt);
    ASSERT_TRUE(seen.has_value());
    EXPECT_EQ(seen.value(), 42);
}

// GpuStage with only CPU passes + an uninitialised GPU: the empty GPU batch is
// skipped, so the CPU passes run with no device access (no Dawn needed).
TEST(GpuStageTest, RunsCpuPassesWithoutDevice) {
    auto gpu = makeUninitialisedGpu();
    PassFlags flags;
    std::vector<std::string> order;

    GpuStage stage{"gpu", gpu};
    stage.add_pass(std::make_unique<FlagPass>("cpu", flags));
    stage.add_pass(std::make_unique<OrderPass>("cpu2", order));

    ASSERT_EQ(stage.initialize(), std::nullopt);
    EXPECT_TRUE(flags.did_initialize);

    ASSERT_EQ(stage.execute(), std::nullopt);
    EXPECT_TRUE(flags.did_execute);
    EXPECT_EQ(order, (std::vector<std::string>{"cpu2"}));
}

TEST(GpuStageTest, StageStopHaltsWithoutDevice) {
    auto gpu = makeUninitialisedGpu();
    std::vector<std::string> order;

    GpuStage stage{"gpu", gpu};
    stage.add_pass(std::make_unique<OrderPass>("before", order));
    stage.add_pass(
        std::make_unique<ReturningPass>("stopper", kStageStopExecution));
    stage.add_pass(std::make_unique<OrderPass>("after", order));

    EXPECT_EQ(stage.execute(), std::nullopt);
    EXPECT_EQ(order, (std::vector<std::string>{"before"}));
}
