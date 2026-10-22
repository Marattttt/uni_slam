#!/bin/bash
# Runs the v2 pipeline (--device=cpu, mapping and loop closure on) over a
# whole UZH-FPV sequence and exports the map, for compare.py. The UZH-FPV
# counterpart of run_wslam_euroc.sh.
#
# Usage: run_wslam_uzh_fpv.sh SEQUENCE_DIR CALIBRATION_DIR OUT_DIR [BUILD_DIR] [-- EXTRA_ARGS...]
#   SEQUENCE_DIR    — unpacked *_snapdragon_with_gt.zip (becomes UZH_FPV_DIR)
#   CALIBRATION_DIR — unpacked *_calib_snapdragon.zip (UZH_FPV_CALIB_DIR)
#   OUT_DIR         — receives map.{ply,json} (plus map_vio.* and friends with
#                     loop closure), run.log, perf_*.yaml and status.txt
#   BUILD_DIR       — build directory holding pc_wslam, configured for the
#                     cameras' 640x480 (default: <repo>/build/release_640x480,
#                     -DWSLAM_FRAME_WIDTH=640 -DWSLAM_FRAME_HEIGHT=480)
#   EXTRA_ARGS are passed to pc_wslam after the defaults.
#
# The models and the other run-time variables come from the repo's .env. The
# pipeline does not stop by itself at the end of the stream, so --max-iters is
# the sequence's left-camera frame count.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ $# -lt 3 ]]; then
    sed -n '2,18p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi

SEQUENCE_DIR="$(realpath "$1")"
CALIBRATION_DIR="$(realpath "$2")"
OUT_DIR="$3"
BUILD_DIR="$REPO_ROOT/build/release_640x480"
shift 3
if [[ $# -gt 0 && "$1" != "--" ]]; then
    BUILD_DIR="$1"
    shift
fi
if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
fi
EXTRA_ARGS=("$@")

BINARY="$BUILD_DIR/pc_wslam"
if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not built" >&2
    exit 1
fi
if [[ ! -f "$SEQUENCE_DIR/left_images.txt" || ! -f "$SEQUENCE_DIR/imu.txt" ]]; then
    echo "error: $SEQUENCE_DIR is not a UZH-FPV text sequence" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(realpath "$OUT_DIR")"

source "$REPO_ROOT/.env"
export UZH_FPV_DIR="$SEQUENCE_DIR"
export UZH_FPV_CALIB_DIR="$CALIBRATION_DIR"
export WSLAM_PERF_OUT_DIR="$OUT_DIR"
export SPDLOG_LEVEL="${SPDLOG_LEVEL:-info}"

FRAMES=$(grep -vc '^#' "$SEQUENCE_DIR/left_images.txt")

echo "binary: $BINARY" > "$OUT_DIR/status.txt"
echo "commit: $(git -C "$REPO_ROOT" describe --always --dirty 2>/dev/null || echo unknown)" >> "$OUT_DIR/status.txt"
echo "sequence: $SEQUENCE_DIR ($FRAMES frames)" >> "$OUT_DIR/status.txt"
echo "calibration: $CALIBRATION_DIR" >> "$OUT_DIR/status.txt"
echo "args: ${EXTRA_ARGS[*]:-}" >> "$OUT_DIR/status.txt"

start=$(date +%s)
exit_code=0
"$BINARY" -v2 --device=cpu --dataset=uzh-fpv --max-iters="$FRAMES" \
    --map-out="$OUT_DIR/map.ply" "${EXTRA_ARGS[@]}" \
    > "$OUT_DIR/run.log" 2>&1 || exit_code=$?
end=$(date +%s)

echo "exit_code: $exit_code" >> "$OUT_DIR/status.txt"
echo "wall_seconds: $((end - start))" >> "$OUT_DIR/status.txt"
cat "$OUT_DIR/status.txt"
exit "$exit_code"
