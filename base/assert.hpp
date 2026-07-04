#pragma once
#include <cstdlib>
#include <format>
#include <print>
#include <source_location>
#include <stacktrace>

#define WSLAM_ASSERT(cond, ...)                                                \
    do {                                                                       \
        /* NOLINTNEXTLINE(readability-simplify-boolean-expr) */                \
        if (!(cond)) [[unlikely]] {                                            \
            ::wslam::base::assert_fail(#cond, std::source_location::current(), \
                                       std::stacktrace::current() __VA_OPT__(  \
                                           , std::format(__VA_ARGS__)));       \
        }                                                                      \
    } while (0)

namespace wslam::base {
[[noreturn]] inline void assert_fail(const char* condition,
                                     std::source_location loc,
                                     const std::stacktrace& trace,
                                     std::string_view message = {}) {
    std::println(stderr, "\n=== ASSERTION FAILED ===");
    std::println(stderr, "Condition : {}", condition);
    std::println(stderr, "Location  : {}:{} in {}", loc.file_name(), loc.line(),
                 loc.function_name());

    if (!message.empty()) {
        std::println(stderr, "Message   : {}", message);
    }

    std::println(stderr, "\nStack trace:\n{}", std::to_string(trace));
    std::abort();
}

};  // namespace wslam::base
