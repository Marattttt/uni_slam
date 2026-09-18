#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "compute/pass.hpp"

using namespace wslam::compute;

namespace {

constexpr std::string kFailureMsg = "womp womp";

// A no-op callback for cases that only inspect construction/return values.
CustomPass::Callback doNothing() {
    return [](auto*) -> std::optional<std::string> { return std::nullopt; };
}

// Minimal concrete Pass returning fixed values, to exercise the base interface
// through a Pass& reference (independent of CustomPass).
class ExecuteValuePass : public Pass {
   public:
    ExecuteValuePass(std::string id, std::optional<std::string> on_execute)
        : id_(std::move(id)), on_execute_(std::move(on_execute)) {}

    [[nodiscard]] std::string getId() const override { return id_; }
    std::optional<std::string> initialize() override { return std::nullopt; }
    std::optional<std::string> execute() override { return on_execute_; }

   private:
    std::string id_;
    std::optional<std::string> on_execute_;
};

}  // namespace

TEST(CustomPassTest, GetIdReturnsConstructorId) {
    CustomPass pass{"myid", doNothing()};
    EXPECT_EQ(pass.getId(), "myid");
}

TEST(CustomPassTest, InitializeReturnsNullopt) {
    CustomPass pass{"init", doNothing()};
    EXPECT_EQ(pass.initialize(), std::nullopt);
}

TEST(CustomPassTest, ExecuteInvokesCallback) {
    bool ran = false;
    CustomPass pass{"run", [&ran](CustomPass*) -> std::optional<std::string> {
                        ran = true;
                        return std::nullopt;
                    }};

    EXPECT_EQ(pass.execute(), std::nullopt);
    EXPECT_TRUE(ran);
}

TEST(CustomPassTest, ExecuteForwardsErrorString) {
    CustomPass pass{"fail", [](CustomPass*) -> std::optional<std::string> {
                        return kFailureMsg;
                    }};

    const auto err = pass.execute();
    ASSERT_TRUE(err.has_value());
    EXPECT_EQ(err.value(), kFailureMsg);
}

TEST(CustomPassTest, CustomPassProvidesItselfToCallback) {
    CustomPass pass{"self", [](CustomPass* self) -> std::optional<std::string> {
                        EXPECT_NE(self, nullptr);
                        if (self != nullptr) {
                            EXPECT_EQ(self->getId(), "self");
                        }
                        return std::nullopt;
                    }};

    EXPECT_EQ(pass.execute(), std::nullopt);
}
