cmake_minimum_required(VERSION 3.16)
project(pc_wslam LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the directory to the search list
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Now you can just use the filename (no .cmake needed)
include(configureDependencies)
configureDependencies()

add_executable(pc_wslam 
    main.cpp
    viz.cpp

    compute/gpu.cpp
    compute/compute.cpp
    compute/pass.cpp
    compute/pass_hellowgsl.cpp
    compute/stage.cpp
    compute/awaiter.cpp

    # data/tum_provider.cpp
    data/euroc_provider.cpp
    data/provider_base.cpp
    data/stb.cpp

    wslam/common.cpp
    wslam/sensor_loader.cpp
    wslam/feature_detect.cpp
    wslam/detect_corners.cpp
    wslam/fill_pyramid.cpp

    base/assert.hpp
 )

target_include_directories(pc_wslam PRIVATE 
    compute
    data
    base
)

target_compile_options(pc_wslam PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
)

option(ENABLE_SANITIZERS TRUE)

if (ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pc_wslam PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )
    target_link_options(pc_wslam PRIVATE
        -fsanitize=address,undefined
    )
endif()

target_link_libraries(pc_wslam 
    PRIVATE
    dawn::webgpu_dawn
    spdlog::spdlog_header_only $<$<BOOL:${MINGW}>:ws2_32> # The bool template comes from spdlog docs
    pango_display pango_python
    stb
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    fast_cpp_csv_parser
)
