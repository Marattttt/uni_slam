#!/usr/bin/env bash

DAWN_DIR="$PWD/vendor/dawn"
BUILD_DIR="$DAWN_DIR/build"
INSTALL_DIR="$DAWN_DIR/install"

rm -r "$BUILD_DIR"
rm -r "$INSTALL_DIR"

set -euo pipefail

cmake -G Ninja -B "$BUILD_DIR" -S "$DAWN_DIR" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

cmake --build "$BUILD_DIR" --parallel $(nproc)

echo "=================================="
echo "Dawn build and install successfull"
echo "=================================="

