add_library(wslam_compute
    src/awaiter.cpp
    src/compute.cpp
    src/gpu.cpp
    src/gpu_stage.cpp
    src/pass.cpp
    src/performance.cpp
    src/stage.cpp
)

add_library(wslam::compute ALIAS wslam_compute)

target_include_directories(wslam_compute PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(LOG_AWAITER_CALLS)
    target_compile_definitions(wslam_compute PRIVATE LOG_AWAITER_CALLS)
endif()

target_link_libraries(wslam_compute
    PUBLIC
    wslam::base
    dawn::webgpu_dawn
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    spdlog::spdlog_header_only $<$<BOOL:${MINGW}>:ws2_32>
)

# CTRE is used in gpu.cpp for find and replace in WGSL source files
target_link_libraries(wslam_compute
    PRIVATE
    ctre::ctre
)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
