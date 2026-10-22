cmake_minimum_required(VERSION 3.24)
project(pc_wslam LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(LOG_AWAITER_CALLS "Enable logging about future adding / completion in compute::Awaiter" OFF)
option(ENABLE_SANITIZERS FALSE)
option(BUILD_TESTS "Build the unit tests (fetches GoogleTest)" OFF)
option(ENABLE_PERF_RECORDING "Record per-pass timings with compute::PerfRecorder (OFF defines NPERF)" ON)
# The camera resolution the GPU buffers, the image pyramid and the model input
# are sized for at compile time (wslam/include/wslam/common.hpp). EuRoC is
# 752x480, UZH-FPV's Snapdragon cameras are 640x480: a dataset of another size
# needs its own build directory
set(WSLAM_FRAME_WIDTH 752 CACHE STRING "Camera image width the pipeline is compiled for")
set(WSLAM_FRAME_HEIGHT 480 CACHE STRING "Camera image height the pipeline is compiled for")

# Declared before configureDependencies() so the GoogleTest fetch inside it sees
# the option's value.
include(configureDependencies)
configureDependencies()

if(BUILD_TESTS)
    enable_testing()
endif()

# Shared warning and sanitize flags for this project's targets
add_library(project_options INTERFACE)
target_compile_options(project_options INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
)

if(ENABLE_SANITIZERS)
    target_compile_options(project_options INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )
    target_link_options(project_options INTERFACE
        -fsanitize=address,undefined
    )
endif()

add_subdirectory(base)
add_subdirectory(compute)
add_subdirectory(data)
add_subdirectory(wslam)

# The base library is header-only so no options needed
target_link_libraries(wslam_compute PRIVATE project_options)
target_link_libraries(wslam_data PRIVATE project_options)
target_link_libraries(wslam_app PRIVATE project_options)
# PUBLIC: common.hpp is a public header and its consumers (tests, pc_wslam)
# must see the same size
target_compile_definitions(wslam_app PUBLIC
    WSLAM_FRAME_WIDTH=${WSLAM_FRAME_WIDTH}
    WSLAM_FRAME_HEIGHT=${WSLAM_FRAME_HEIGHT}
)

add_executable(pc_wslam main.cpp)
target_link_libraries(pc_wslam PRIVATE wslam::app onnxruntime::onnxruntime project_options)
