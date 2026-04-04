#pragma once
#include <cstdlib>
#include <format>
#include <print>
#include <source_location>
#include <stacktrace>

#define WSLAM_ASSERT(cond, ...)                                   \
    do {                                                          \
        if (!(cond)) [[unlikely]] {                               \
            auto assert_loc = std::source_location::current();    \
            auto assert_trace = std::stacktrace::current();       \
            [&] {                                                 \
                if constexpr (sizeof(#__VA_ARGS__) > 1)           \
                    assert_fail(#cond, assert_loc, assert_trace,  \
                                std::format(__VA_ARGS__));        \
                else                                              \
                    assert_fail(#cond, assert_loc, assert_trace); \
            }();                                                  \
        }                                                         \
    } while (0)

namespace wslam::base {
inline void assert_fail(const char* condition, std::source_location loc,
                        const std::stacktrace& trace,
                        std::string_view message = {}) {
    std::println(stderr, "\n=== ASSERTION FAILED ===");
    std::println(stderr, "Condition : {}", condition);
    std::println(stderr, "Location  : {}:{} in {}", loc.file_name(), loc.line(),
                 loc.function_name());

    if (!message.empty()) std::println(stderr, "Message   : {}", message);

    std::println(stderr, "\nStack trace:\n{}", std::to_string(trace));
    std::abort();
}

};  // namespace wslam::base
