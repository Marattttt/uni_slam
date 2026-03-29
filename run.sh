#!/bin/bash

mkdir build &> /dev/null

set -oue pipefail

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build

echo "BUILD FOR PROJECT pc_wslam SUCCESSFUL!!!"

export WSLAM_SHADER_SRC_DIR="$PWD"

source .env

./build/pc_wslam
