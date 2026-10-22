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
  -v2               Run the v2 pipeline.
  --dataset=euroc|uzh-fpv
                    Dataset format (default euroc: EUROC_DIR). uzh-fpv reads
                    UZH_FPV_DIR and UZH_FPV_CALIB_DIR and needs a 640x480
                    build: pass 640x480 as well.
  640x480           Build into build/640x480 for 640x480 cameras
                    (UZH-FPV Snapdragon) instead of build/ (752x480, EuRoC).
  --device=gpu|cpu  Where compute runs (default gpu). cpu runs Dawn on its
                    SwiftShader CPU adapter and, with -v2, the model on
                    ONNX Runtime's CPU execution provider.
  awaiter           Enable verbose awaiter logging (LOG_AWAITER_CALLS=ON).
  --max-iters=N     Stop after N top-level pipeline executions (0 = unlimited).
  --map-out=PATH    Write the final factor-graph map to PATH. Must end in
                    .ply; a sibling .json metadata file is derived
                    automatically.
  --keyframes=PATH  Only feed the frames listed in this keyframe schedule
                    JSON to the pipeline (all IMU is kept). Requires -v2.
  -no-mapping       v2 only: run without the mapping stage; the model then
                    pairs consecutive frames.
  -no-loop-closure  v2 only: run the mapping stage without loop closure.
  -no-frame-prefilter
                    v2 with mapping only: run the model on every frame
                    instead of skipping those the IMU predicts the
                    keyframe gate would reject.
  --record-frontend=PATH
                    v2 only: record the front-end output (model matches,
                    IMU, keyframe decisions) to PATH (.msgpack) for
                    replaying the mapping stage without the GPU.
EOF
}

LOG_AWAITER=OFF
BUILD_DIR=build
FRAME_SIZE_ARGS=()
RUN_ARGS=()
BUILD_TYPE=Debug

for arg in "$@"; do
    case "$arg" in
        -h|--help)      print_help; exit 0 ;;
        awaiter)        LOG_AWAITER=ON ;;
        -gui)           RUN_ARGS+=("-gui") ;;
        -v2)            RUN_ARGS+=("-v2") ;;
        --device=*)     RUN_ARGS+=("$arg") ;;
        --dataset=*)    RUN_ARGS+=("$arg") ;;
        640x480)        BUILD_DIR=build/640x480
                        FRAME_SIZE_ARGS=(-DWSLAM_FRAME_WIDTH=640 -DWSLAM_FRAME_HEIGHT=480) ;;
        --max-iters=*)  RUN_ARGS+=("$arg") ;;
        --map-out=*)    RUN_ARGS+=("$arg") ;;
        --keyframes=*)  RUN_ARGS+=("$arg") ;;
        -no-mapping)    RUN_ARGS+=("$arg") ;;
        -no-loop-closure) RUN_ARGS+=("$arg") ;;
        -no-frame-prefilter) RUN_ARGS+=("$arg") ;;
        --record-frontend=*) RUN_ARGS+=("$arg") ;;
        Debug)          BUILD_TYPE="Debug" ;;
        Release)        BUILD_TYPE="RelWithDebInfo" ;;
        *)              echo "error: unknown argument '$arg'" >&2
                        print_help >&2
                        exit 1 ;;
    esac
done

mkdir -p "$BUILD_DIR"

cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLOG_AWAITER_CALLS=$LOG_AWAITER \
    "${FRAME_SIZE_ARGS[@]}"

cmake --build "$BUILD_DIR"

echo "BUILD FOR PROJECT pc_wslam SUCCESSFUL!!!"

export WSLAM_SHADER_SRC_DIR="$PWD"

source .env
"./$BUILD_DIR/pc_wslam" "${RUN_ARGS[@]}"
