#!/bin/bash
# Runs the v2 pipeline (--device=cpu, mapping and loop closure on) over a
# whole EuRoC sequence and exports the map, for compare.py.
#
# Usage: run_wslam_euroc.sh MAV0_DIR OUT_DIR [BUILD_DIR] [-- EXTRA_ARGS...]
#   MAV0_DIR  — the sequence's mav0/ directory (becomes EUROC_DIR)
#   OUT_DIR   — receives map.{ply,json} (plus map_vio.* and friends with
#               loop closure), run.log, perf_*.yaml and status.txt
#   BUILD_DIR — build directory holding pc_wslam (default: <repo>/build)
#   EXTRA_ARGS are passed to pc_wslam after the defaults.
#
# The models and the other run-time variables come from the repo's .env;
# EUROC_DIR is overridden. The pipeline does not stop by itself at the end
# of the stream, so --max-iters is the sequence's cam0 frame count.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ $# -lt 2 ]]; then
    sed -n '2,13p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi

MAV0_DIR="$(realpath "$1")"
OUT_DIR="$2"
BUILD_DIR="$REPO_ROOT/build"
shift 2
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
if [[ ! -f "$MAV0_DIR/cam0/data.csv" ]]; then
    echo "error: $MAV0_DIR is not a EuRoC mav0/ directory" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(realpath "$OUT_DIR")"

source "$REPO_ROOT/.env"
export EUROC_DIR="$MAV0_DIR"
export WSLAM_PERF_OUT_DIR="$OUT_DIR"
export SPDLOG_LEVEL="${SPDLOG_LEVEL:-info}"

FRAMES=$(grep -vc '^#' "$MAV0_DIR/cam0/data.csv")

echo "binary: $BINARY" > "$OUT_DIR/status.txt"
echo "commit: $(git -C "$BUILD_DIR" describe --always --dirty 2>/dev/null || echo unknown)" >> "$OUT_DIR/status.txt"
echo "sequence: $MAV0_DIR ($FRAMES frames)" >> "$OUT_DIR/status.txt"
echo "args: ${EXTRA_ARGS[*]:-}" >> "$OUT_DIR/status.txt"

start=$(date +%s)
exit_code=0
"$BINARY" -v2 --device=cpu --max-iters="$FRAMES" \
    --map-out="$OUT_DIR/map.ply" "${EXTRA_ARGS[@]}" \
    > "$OUT_DIR/run.log" 2>&1 || exit_code=$?
end=$(date +%s)

echo "exit_code: $exit_code" >> "$OUT_DIR/status.txt"
echo "wall_seconds: $((end - start))" >> "$OUT_DIR/status.txt"
cat "$OUT_DIR/status.txt"
exit "$exit_code"
