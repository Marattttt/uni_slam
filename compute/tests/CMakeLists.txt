add_executable(compute_tests
    stage_tests.cpp
)

target_link_libraries(compute_tests
    PRIVATE
    wslam::compute
    GTest::gtest_main
    project_options
)

include(GoogleTest)
gtest_discover_tests(compute_tests PROPERTIES LABELS compute)
