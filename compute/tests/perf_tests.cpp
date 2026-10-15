#include <gtest/gtest.h>

#include <string>
#include <thread>

#include "compute/performance.hpp"

using namespace wslam::compute;

namespace {

size_t CallCount(const YAML::Node& node) {
    return node["stats"]["count"].as<size_t>();
}

#ifdef NPERF
#define SKIP_WITHOUT_RECORDING() GTEST_SKIP() << "built with NPERF"
#else
#define SKIP_WITHOUT_RECORDING() static_cast<void>(0)
#endif

TEST(PerfRecorderTest, ThreadsRecordUnderTheirOwnRoot) {
    SKIP_WITHOUT_RECORDING();
    PerfRecorder perf;

    const auto record = [&perf](const std::string& thread_id) {
        PerfRecorder::setThreadId(thread_id);
        for (int i = 0; i < 3; ++i) {
            const auto outer = perf.beginRecord("outer");
            const auto inner = perf.beginRecord("inner");
        }
    };

    std::thread first(record, "first");
    std::thread second(record, "second");
    first.join();
    second.join();

    const YAML::Node yaml = perf.toYaml();
    ASSERT_EQ(yaml.size(), 2U);
    for (const char* thread_id : {"first", "second"}) {
        const YAML::Node root = yaml[thread_id];
        ASSERT_TRUE(root) << thread_id;
        EXPECT_EQ(CallCount(root["outer"]), 3U);
        EXPECT_EQ(CallCount(root["outer"]["inner"]), 3U);
    }
}

TEST(PerfRecorderTest, UnnamedThreadGetsADefaultRoot) {
    SKIP_WITHOUT_RECORDING();
    PerfRecorder perf;
    std::string thread_id;

    std::thread thread([&] {
        const auto scope = perf.beginRecord("work");
        thread_id = PerfRecorder::getThreadId();
    });
    thread.join();

    EXPECT_TRUE(thread_id.starts_with("thread "));
    EXPECT_EQ(CallCount(perf.toYaml()[thread_id]["work"]), 1U);
}

TEST(PerfRecorderTest, ScopeClosingAfterClearRecordsAlone) {
    SKIP_WITHOUT_RECORDING();
    PerfRecorder perf;
    std::thread thread([&] {
        PerfRecorder::setThreadId("worker");
        const auto outer = perf.beginRecord("outer");
        { const auto inner = perf.beginRecord("inner"); }
        perf.clear();
    });
    thread.join();

    const YAML::Node yaml = perf.toYaml();
    EXPECT_EQ(CallCount(yaml["worker"]["outer"]), 1U);
    EXPECT_FALSE(yaml["worker"]["outer"]["inner"]);
}

}  // namespace
