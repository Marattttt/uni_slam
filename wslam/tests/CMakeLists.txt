add_executable(wslam_tests
    epipolar_geometry_tests.cpp
    keyframe_schedule_tests.cpp
    match_scores_tests.cpp
    sensor_loader_tests.cpp
    local_smoother_tests.cpp
    loop_closure_tests.cpp
    loop_closure_dataset_tests.cpp
    mapping_replay_tests.cpp
    motion_prediction_tests.cpp
    pose_estimate_dataset_tests.cpp
)

# The tests reach the library's internal headers by the same bare names its
# own sources use
target_include_directories(wslam_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/wslam/include/wslam
    ${PROJECT_SOURCE_DIR}/wslam/src
)

target_compile_definitions(wslam_tests PRIVATE
    WSLAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_link_libraries(wslam_tests
    PRIVATE
    wslam::app
    # wslam_app links GTSAM privately; the mapping tests use it directly
    gtsam
    GTest::gtest_main
    project_options
)

include(GoogleTest)
gtest_discover_tests(wslam_tests PROPERTIES LABELS wslam)
