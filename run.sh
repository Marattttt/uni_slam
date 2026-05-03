#!/bin/bash
set -euo pipefail

mkdir -p build

LOG_AWAITER=OFF
if [ "${1:-}" = "awaiter" ]; then
    LOG_AWAITER=ON
fi

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLOG_AWAITER_CALLS=$LOG_AWAITER

cmake --build build

echo "BUILD FOR PROJECT pc_wslam SUCCESSFUL!!!"

export WSLAM_SHADER_SRC_DIR="$PWD"

source .env
./build/pc_wslam
