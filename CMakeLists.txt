cmake_minimum_required(VERSION 3.16)
project(pc_wslam LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(configureDependencies)
configureDependencies()

option(LOG_AWAITER_CALLS "Enable logging about future adding / completion in compute::Awaiter" OFF)
option(ENABLE_SANITIZERS TRUE)

add_executable(pc_wslam 
    main.cpp

    compute/awaiter.cpp
    compute/compute.cpp
    compute/gpu.cpp
    compute/pass.cpp
    compute/pass_hellowgsl.cpp
    compute/stage.cpp

    data/euroc_provider.cpp
    data/provider_base.cpp
    data/stb.cpp
    # data/tum_provider.cpp

    wslam/common.cpp
    wslam/cull_corners.cpp
    wslam/detect_corners.cpp
    wslam/feature_detect.cpp
    wslam/fill_pyramid.cpp
    wslam/generate_features.cpp
    wslam/load_features_cpu.cpp
    wslam/match_features_cpu.cpp
    wslam/pose_estimate_cpu.cpp
    wslam/sensor_loader.cpp
    wslam/viz.cpp
    wslam/vizualize_data.cpp
    wslam/wslam.hpp
 )

target_include_directories(pc_wslam PRIVATE 
    compute
    data
    base
)


target_compile_options(pc_wslam PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
)

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

if(LOG_AWAITER_CALLS) 
    target_compile_definitions(pc_wslam PRIVATE LOG_AWAITER_CALLS)
endif()

target_link_libraries(pc_wslam 
    PRIVATE
    dawn::webgpu_dawn
    spdlog::spdlog_header_only $<$<BOOL:${MINGW}>:ws2_32> # The bool template comes from spdlog docs
    pango_display pango_python
    stb
    ctre::ctre
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    fast_cpp_csv_parser
)
