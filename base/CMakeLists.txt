add_library(wslam_base INTERFACE)
add_library(wslam::base ALIAS wslam_base)

target_include_directories(wslam_base INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# base/assert.hpp's WSLAM_ASSERT uses std::stacktrace, which on GCC lives in
# libstdc++exp; consumers of base inherit the link requirement.
target_link_libraries(wslam_base INTERFACE
    $<$<CXX_COMPILER_ID:GNU>:stdc++exp>
)
