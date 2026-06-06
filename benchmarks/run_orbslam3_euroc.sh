#!/bin/bash
# Runs the vendored ORB-SLAM3 in mono-inertial mode on the EuRoC sequence
# pointed at by EUROC_DIR (from the environment, or the repo's .env as a
# fallback; V1_01_easy). Trajectory files (f_*.txt full trajectory,
# kf_*.txt keyframes; both TUM-ordered with nanosecond timestamps) land in
# benchmarks/results/.
#
# Usage: run_orbslam3_euroc.sh [MAX_FRAMES]
#   MAX_FRAMES — optional: process only the first N images of the
#   sequence. ORB-SLAM3's EuRoC examples iterate over exactly the images
#   listed in the timestamps file, so a truncated copy of that file IS the
#   frame limiter — no source patch needed. The output is suffixed
#   _<N>f so partial runs never overwrite the full baseline.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ORB_DIR="$SCRIPT_DIR/ORB_SLAM3"
RESULTS_DIR="$SCRIPT_DIR/results"
RUN_NAME="dataset-V101_monoi"
TIMESTAMPS_FILE="$ORB_DIR/Examples/Monocular-Inertial/EuRoC_TimeStamps/V101.txt"

MAX_FRAMES="${1:-}"

# EUROC_DIR (the mav0/ root) may already be exported; otherwise pull it
# from the repo .env. ORB-SLAM3's example expects the directory that
# *contains* mav0/.
if [[ -z "${EUROC_DIR:-}" ]]; then
    source "$REPO_ROOT/.env"
fi
SEQ_DIR="$(dirname "$EUROC_DIR")"

BINARY="$ORB_DIR/Examples/Monocular-Inertial/mono_inertial_euroc"
if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not built — run benchmarks/build_orbslam3.sh first" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Frame limiting: hand the binary a truncated timestamps file.
if [[ -n "$MAX_FRAMES" ]]; then
    if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ ]]; then
        echo "error: MAX_FRAMES must be a positive integer, got '$MAX_FRAMES'" >&2
        exit 1
    fi
    truncated="$RESULTS_DIR/V101_first${MAX_FRAMES}.txt"
    head -n "$MAX_FRAMES" "$TIMESTAMPS_FILE" > "$truncated"
    TIMESTAMPS_FILE="$truncated"
    RUN_NAME="${RUN_NAME}_${MAX_FRAMES}f"
    echo "Limiting run to the first $MAX_FRAMES frames ($truncated)"
fi

# The example writes f_/kf_ trajectory files into the CWD, so run from the
# results directory.
#
# ORB-SLAM3 has a long-standing shutdown race (viewer thread still inside
# cv::waitKey while the main thread tears down the GL context) that segfaults
# *after* both trajectory files are saved. Don't let that abort the script;
# judge success by the presence of non-empty trajectory files instead.
cd "$RESULTS_DIR"
exit_code=0
"$BINARY" \
    "$ORB_DIR/Vocabulary/ORBvoc.txt" \
    "$ORB_DIR/Examples/Monocular-Inertial/EuRoC.yaml" \
    "$SEQ_DIR" \
    "$TIMESTAMPS_FILE" \
    "$RUN_NAME" || exit_code=$?

if [[ "$exit_code" -ne 0 ]]; then
    echo "warning: mono_inertial_euroc exited with code $exit_code" \
         "(known upstream shutdown crash; checking trajectory files anyway)" >&2
fi

for trajectory in "$RESULTS_DIR/f_$RUN_NAME.txt" "$RESULTS_DIR/kf_$RUN_NAME.txt"; do
    if [[ ! -s "$trajectory" ]]; then
        echo "error: expected trajectory file '$trajectory' is missing or empty" >&2
        exit 1
    fi
done

echo "Trajectories written to:"
ls -la "$RESULTS_DIR/f_$RUN_NAME.txt" "$RESULTS_DIR/kf_$RUN_NAME.txt"
