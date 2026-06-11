#!/bin/bash
set -euo pipefail

print_help() {
    cat <<EOF
Usage: $(basename "$0") [options]

Configures, builds, and runs the pc_wslam binary against the dataset
referenced in .env.

Options:
  -h, --help        Show this help and exit.
  -gui              Enable the Pangolin visualization GUI.
  awaiter           Enable verbose awaiter logging (LOG_AWAITER_CALLS=ON).
  --max-iters=N     Stop after N top-level pipeline executions (0 = unlimited).
  --map-out=PATH    Write the final factor-graph map to PATH. Must end in
                    .ply; a sibling .json metadata file is derived
                    automatically.
EOF
}

mkdir -p build

LOG_AWAITER=OFF
RUN_ARGS=()
BUILD_TYPE=Debug

for arg in "$@"; do
    case "$arg" in
        -h|--help)      print_help; exit 0 ;;
        awaiter)        LOG_AWAITER=ON ;;
        -gui)           RUN_ARGS+=("-gui") ;;
        --max-iters=*)  RUN_ARGS+=("$arg") ;;
        --map-out=*)    RUN_ARGS+=("$arg") ;;
        Debug)          BUILD_TYPE="Debug" ;;
        Release)        BUILD_TYPE="Release" ;;
        *)              echo "error: unknown argument '$arg'" >&2
                        print_help >&2
                        exit 1 ;;
    esac
done

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLOG_AWAITER_CALLS=$LOG_AWAITER

cmake --build build

echo "BUILD FOR PROJECT pc_wslam SUCCESSFUL!!!"

export WSLAM_SHADER_SRC_DIR="$PWD"

source .env
./build/pc_wslam "${RUN_ARGS[@]}"
