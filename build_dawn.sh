#!/usr/bin/env bash

BUILD_TYPE=Release

DAWN_DIR="$PWD/vendor/dawn"
BUILD_DIR="$DAWN_DIR/build/$BUILD_TYPE"
INSTALL_DIR="$DAWN_DIR/install/$BUILD_TYPE"


rm -r "$BUILD_DIR"
rm -r "$INSTALL_DIR"

set -euo pipefail

cmake \
	-G Ninja \
	-B "$BUILD_DIR" \
	-S "$DAWN_DIR" \
	-D CMAKE_INSTALL_PREFIX="$INSTALL_DIR"\
	-D DAWN_FETCH_DEPENDENCIES=ON \
	-D DAWN_ENABLE_INSTALL=ON \
	-D CMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" --parallel $(nproc)

cmake --install "$BUILD_DIR"

echo "=================================="
echo "Dawn build and install successfull"
echo "=================================="

