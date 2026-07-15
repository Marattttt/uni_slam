cmake_minimum_required(VERSION 3.24)
project(pc_wslam LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(configureDependencies)
configureDependencies()

option(LOG_AWAITER_CALLS "Enable logging about future adding / completion in compute::Awaiter" OFF)
option(ENABLE_SANITIZERS FALSE)

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

add_executable(pc_wslam main.cpp)
target_link_libraries(pc_wslam PRIVATE wslam::app project_options)
