add_library(wslam_app
    src/common.cpp
    src/wslam.cpp
    src/export.cpp
    src/keyframe_schedule.cpp
    src/frontend_recording.cpp

    src/feature_detect/cull_corners.cpp
    src/feature_detect/detect_corners.cpp
    src/feature_detect/feature_detect.cpp
    src/feature_detect/fill_pyramid.cpp
    src/feature_detect/generate_features.cpp
    src/feature_detect/sensor_loader.cpp

    src/feature_detect_v2/feature_detect.cpp
    src/feature_detect_v2/fill_pyramid.cpp
    src/feature_detect_v2/onnx_feature_detect.cpp
    src/feature_detect_v2/pack_model_input.cpp
    src/feature_detect_v2/preserve_pyramid.cpp
    src/feature_detect_v2/sensor_loader.cpp
    src/feature_detect_v2/shared_bindings.cpp

    src/pose_estimate_cpu/load_features_cpu.cpp
    src/pose_estimate_cpu/match_features_cpu.cpp
    src/pose_estimate_cpu/pose_estimate_cpu.cpp
    src/pose_estimate_cpu/ransac_cpu.cpp
    src/pose_estimate_cpu/triangulate_cpu.cpp

    src/pose_estimate_v2/load_matches.cpp
    src/pose_estimate_v2/match_scores.cpp
    src/pose_estimate_v2/pose_estimate.cpp
    src/pose_estimate_v2/ransac.cpp
    src/pose_estimate_v2/triangulate.cpp
    src/pose_estimate_v2/util/epipolar_geometry.cpp

    src/map/factor_builder.cpp
    src/map/isam2_update.cpp
    src/map/isam2_worker.cpp
    src/map/keyframe_gate.cpp
    src/map/map_build_factors.cpp
    src/map/map_filter_keyframe.cpp
    src/map/map_submit_updates.cpp
    src/map/map_stage.cpp
    src/map/utils.cpp
    src/map/mapping.cpp

    src/map_v2/associate_landmarks.cpp
    src/map_v2/build_factors.cpp
    src/map_v2/filter_keyframe.cpp
    src/map_v2/map_stage.cpp
    src/map_v2/smoother_worker.cpp
    src/map_v2/submit_updates.cpp

    src/viz/viz.cpp
    src/viz/vizualize_data.cpp
)

add_library(wslam::app ALIAS wslam_app)

target_include_directories(wslam_app PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Private headers used internally by bare name: common.hpp (include/wslam),
# models.hpp / mapping_state.hpp / export.hpp (src). Sibling headers within a
# feature subdir resolve via same-directory lookup.
target_include_directories(wslam_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/wslam
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(wslam_app
    PUBLIC
    wslam::compute
    wslam::data
    wslam::base
)

target_link_libraries(wslam_app
    PRIVATE
    gtsam
    pango_display pango_python
    nlohmann_json::nlohmann_json
    ctre::ctre
    spdlog::spdlog_header_only $<$<BOOL:${MINGW}>:ws2_32>
)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
