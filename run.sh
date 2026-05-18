#!/bin/bash
set -euo pipefail

mkdir -p build

LOG_AWAITER=OFF
RUN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        awaiter)        LOG_AWAITER=ON ;;
        -gui)           RUN_ARGS+=("-gui") ;;
        --max-iters=*)  RUN_ARGS+=("$arg") ;;
    esac
done

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLOG_AWAITER_CALLS=$LOG_AWAITER

cmake --build build

echo "BUILD FOR PROJECT pc_wslam SUCCESSFUL!!!"

export WSLAM_SHADER_SRC_DIR="$PWD"

source .env
./build/pc_wslam "${RUN_ARGS[@]}"
