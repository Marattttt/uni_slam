#!/bin/bash
# Runs the vendored ORB-SLAM3 in mono-inertial mode on a UZH-FPV sequence,
# through the EuRoC-layout mirror benchmarks/uzh_fpv_to_euroc.py writes (the
# example binary is mono_inertial_euroc; ORB-SLAM3 stays unpatched).
# Trajectory files (f_*.txt every frame, kf_*.txt keyframes; IMU poses,
# TUM-ordered, nanosecond timestamps on the IMU clock) land in
# benchmarks/results/ as f_<NAME>_monoi[_r<I>].txt.
#
# Usage: [ORB_SETTINGS=file.yaml] run_orbslam3_uzh_fpv.sh MIRROR_DIR NAME [RUN_INDEX]
#   MIRROR_DIR — uzh_fpv_to_euroc.py's OUT_DIR (mav0/, timestamps.txt,
#                orbslam3_monoi.yaml)
#   NAME       — run name, e.g. uzh-indoor_45_2
#   RUN_INDEX  — optional: suffixes the output _r<N>, for repeated runs
#                (ORB-SLAM3 is not deterministic)
#   ORB_SETTINGS — settings file name inside MIRROR_DIR (default
#                orbslam3_monoi.yaml; uzh_fpv_to_euroc.py --settings-name)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ORB_DIR="$SCRIPT_DIR/ORB_SLAM3"
RESULTS_DIR="$SCRIPT_DIR/results"

if [[ $# -lt 2 ]]; then
    sed -n '2,17p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi
MIRROR_DIR="$(realpath "$1")"
RUN_NAME="${2}_monoi"
RUN_INDEX="${3:-}"
SETTINGS="$MIRROR_DIR/${ORB_SETTINGS:-orbslam3_monoi.yaml}"
if [[ -n "$RUN_INDEX" ]]; then
    RUN_NAME="${RUN_NAME}_r${RUN_INDEX}"
fi

for required in "$MIRROR_DIR/mav0/imu0/data.csv" "$MIRROR_DIR/timestamps.txt" \
                "$SETTINGS"; do
    if [[ ! -f "$required" ]]; then
        echo "error: $required missing — run benchmarks/uzh_fpv_to_euroc.py first" >&2
        exit 1
    fi
done

BINARY="$ORB_DIR/Examples/Monocular-Inertial/mono_inertial_euroc"
if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not built — run benchmarks/build_orbslam3.sh first" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# See run_orbslam3_euroc.sh: Pangolin's libraries find each other here
export LD_LIBRARY_PATH="$REPO_ROOT/vendor/pangolin/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The example writes its trajectories into the CWD. It segfaults at shutdown
# after saving them (upstream viewer teardown race), so success is judged by
# the files
cd "$RESULTS_DIR"
exit_code=0
"$BINARY" \
    "$ORB_DIR/Vocabulary/ORBvoc.txt" \
    "$SETTINGS" \
    "$MIRROR_DIR" \
    "$MIRROR_DIR/timestamps.txt" \
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
