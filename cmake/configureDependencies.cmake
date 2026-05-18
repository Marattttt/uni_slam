function (configureDependencies) 
    include(FetchContent)

    find_package(Threads REQUIRED)

    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.15.1
        GIT_SHALLOW    TRUE
    )
    # Use the header only version
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_USE_STD_FORMAT ON  CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)

    # Remove compiler warnings 
    get_target_property(SPDLOG_REAL spdlog::spdlog_header_only ALIASED_TARGET)
    get_target_property(SPDLOG_INC spdlog::spdlog_header_only INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${SPDLOG_REAL} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${SPDLOG_INC}"
    )

    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        5.0.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)

    # Remove compiler warnings
    get_target_property(EIGEN_REAL Eigen3::Eigen ALIASED_TARGET)
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${EIGEN_REAL} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${EIGEN_INC}"
    )

    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(stb)
    add_library(stb INTERFACE)
    target_include_directories(stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")

    FetchContent_Declare(
        ctre
        GIT_REPOSITORY https://github.com/hanickadot/compile-time-regular-expressions.git
        GIT_TAG        v3.10.0
    )
    FetchContent_MakeAvailable(ctre)

    FetchContent_Declare(
        nlohmann_json
        URL      https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install    OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)
    # suppress warnings
    get_target_property(JSON_REAL nlohmann_json::nlohmann_json ALIASED_TARGET)
    get_target_property(JSON_INC  nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${JSON_REAL} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${JSON_INC}"
    )

    FetchContent_Declare(
      yaml-cpp
      GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
      GIT_TAG yaml-cpp-0.9.0
    )
    FetchContent_MakeAvailable(yaml-cpp)

    FetchContent_Declare(
        fast_cpp_csv_parser
        GIT_REPOSITORY https://github.com/ben-strasser/fast-cpp-csv-parser.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(fast_cpp_csv_parser)
    add_library(fast_cpp_csv_parser INTERFACE)
    target_include_directories(fast_cpp_csv_parser SYSTEM INTERFACE
        "${fast_cpp_csv_parser_SOURCE_DIR}"
    )

    # FetchContent_Declare(
    #     pangolin
    #     GIT_REPOSITORY "https://github.com/stevenlovegrove/Pangolin"
    #     GIT_TAG        v0.9.4
    #     GIT_SHALLOW    TRUE
    # )
    # FetchContent_MakeAvailable(pangolin)

    # Dawn has a complex, multi-step build process with its own dependency
    # fetching (depot_tools/gclient). It is much more reliable to build it
    # separately and point CMake at the install tree.
    set(Dawn_DIR "${PROJECT_SOURCE_DIR}/vendor/dawn/install/Debug/lib/cmake/Dawn")
    find_package(Dawn REQUIRED)

    # Pangolin has many optional system dependencies (OpenGL, GLEW, libjpeg, etc.)
    # that are awkward to manage through FetchContent, and it doesn't always
    # install cleanly as a sub-project. Pre-building it is more predictable.
    set(Pangolin_DIR "${PROJECT_SOURCE_DIR}/vendor/pangolin/install/lib/cmake/Pangolin")
    find_package(Pangolin 0.8 REQUIRED)
    include_directories(${Pangolin_INCLUDE_DIRS})

    # Why not vendor it?
    set(GTSAM_DIR "${PROJECT_SOURCE_DIR}/vendor/gtsam/install/lib/cmake/GTSAM")
    find_package(GTSAM REQUIRED)
endfunction()
