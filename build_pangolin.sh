#!/usr/bin/env bash

PANGO_DIR="$PWD/vendor/pangolin"
BUILD_DIR="$PANGO_DIR/build"
INSTALL_DIR="$PANGO_DIR/install"

# Install dependencies (as described above, or your preferred method)
./scripts/install_prerequisites.sh recommended

# Configure and build
cmake -B build
cmake --build build

# with Ninja for faster builds (sudo apt install ninja-build)
cmake -B build -GNinja
cmake --build build

# GIVEME THE PYTHON STUFF!!!! (Check the output to verify selected python version)
cmake --build build -t pypangolin_pip_install

# Run me some tests! (Requires Catch2 which must be manually installed on Ubuntu.)
cmake -B build -G Ninja -D BUILD_TESTS=ON
cmake --build build
cd build
ctest
