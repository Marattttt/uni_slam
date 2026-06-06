#!/bin/bash
# Builds the vendored ORB-SLAM3 (benchmarks/ORB_SLAM3) against the system
# toolchain. Mirrors ORB_SLAM3/build.sh (DBoW2 -> g2o -> Sophus -> vocabulary
# -> main library + examples) but adds the flags a modern setup needs:
#   - CMAKE_POLICY_VERSION_MINIMUM=3.5: the vendored CMakeLists predate the
#     CMake 4 minimum-version floor.
#   - CMAKE_CXX_STANDARD=17 for the Thirdparty builds that do not pin a
#     standard themselves (the main CMakeLists is patched to C++17 directly,
#     which Pangolin 0.9 headers require).
#   - CMAKE_PREFIX_PATH pointing at the project's prebuilt Pangolin so no
#     system-wide Pangolin install is needed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ORB_DIR="$SCRIPT_DIR/ORB_SLAM3"
PANGOLIN_PREFIX="$REPO_ROOT/vendor/pangolin/install"

COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

echo "== Configuring and building Thirdparty/DBoW2 =="
cmake -S "$ORB_DIR/Thirdparty/DBoW2" -B "$ORB_DIR/Thirdparty/DBoW2/build" \
    "${COMMON_FLAGS[@]}" -DCMAKE_CXX_STANDARD=17
cmake --build "$ORB_DIR/Thirdparty/DBoW2/build" -j"$(nproc)"

echo "== Configuring and building Thirdparty/g2o =="
cmake -S "$ORB_DIR/Thirdparty/g2o" -B "$ORB_DIR/Thirdparty/g2o/build" \
    "${COMMON_FLAGS[@]}" -DCMAKE_CXX_STANDARD=17
cmake --build "$ORB_DIR/Thirdparty/g2o/build" -j"$(nproc)"

echo "== Configuring and building Thirdparty/Sophus =="
cmake -S "$ORB_DIR/Thirdparty/Sophus" -B "$ORB_DIR/Thirdparty/Sophus/build" \
    "${COMMON_FLAGS[@]}" -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build "$ORB_DIR/Thirdparty/Sophus/build" -j"$(nproc)"

echo "== Uncompressing vocabulary =="
if [[ ! -f "$ORB_DIR/Vocabulary/ORBvoc.txt" ]]; then
    tar -xf "$ORB_DIR/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB_DIR/Vocabulary"
fi

echo "== Configuring and building ORB_SLAM3 =="
cmake -S "$ORB_DIR" -B "$ORB_DIR/build" \
    "${COMMON_FLAGS[@]}" \
    -DCMAKE_PREFIX_PATH="$PANGOLIN_PREFIX"
cmake --build "$ORB_DIR/build" -j"$(nproc)"

echo "ORB-SLAM3 BUILD SUCCESSFUL"
