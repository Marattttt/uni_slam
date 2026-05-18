#!/usr/bin/env bash

GTSAM_DIR="$PWD/vendor/gtsam"
BUILD_DIR="$GTSAM_DIR/build"
INSTALL_DIR="$GTSAM_DIR/install"

rm -rf "$BUILD_DIR"
rm -rf "$INSTALL_DIR"

set -euo pipefail

cmake \
	-B "$BUILD_DIR" \
	-G Ninja \
	-S "$GTSAM_DIR" \
	-D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
	-D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
	-D CMAKE_BUILD_TYPE=Debug \
	-D GTSAM_BUILD_TESTS=OFF \
	-D GTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
	-D GTSAM_BUILD_TIMING_ALWAYS=OFF \
	-D GTSAM_BUILD_PYTHON=OFF \
	-D GTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
	-D GTSAM_WITH_TBB=OFF \
	-D GTSAM_BUILD_UNSTABLE=OFF

cmake --build "$BUILD_DIR" --parallel $(nproc)
cmake --install "$BUILD_DIR"

echo "=================================="
echo "GTSAM build and install successful"
echo "=================================="
