add_library(wslam_data
    src/euroc_provider.cpp
    src/provider_base.cpp
    src/stb.cpp
)

add_library(wslam::data ALIAS wslam_data)

target_include_directories(wslam_data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(wslam_data
    PUBLIC
    Eigen3::Eigen                 # provider_base.hpp exposes Eigen types
)

target_link_libraries(wslam_data
    PRIVATE
    spdlog::spdlog_header_only $<$<BOOL:${MINGW}>:ws2_32>
    yaml-cpp::yaml-cpp
    fast_cpp_csv_parser
    stb
)
