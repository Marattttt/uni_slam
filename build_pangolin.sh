#!/usr/bin/env bash

PANGO_DIR="$PWD/vendor/pangolin"
BUILD_DIR="$PANGO_DIR/build"
INSTALL_DIR="$PANGO_DIR/install"

rm -r "$BUILD_DIR"
rm -r "$INSTALL_DIR"

mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

set -oue pipefail

cmake \
	-B "$BUILD_DIR" \
	-G Ninja \
	-S "$PANGO_DIR" \
	-D BUILD_TOOLS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D BUILD_ASAN=ON \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_DISABLE_FIND_PACKAGE_TIFF=ON \
	-D CMAKE_INSTALL_PREFIX="$INSTALL_DIR"
	
cmake --build "$BUILD_DIR" --parallel $(nproc)
cmake --install "$BUILD_DIR"
