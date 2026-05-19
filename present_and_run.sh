#!/bin/bash
set -euo pipefail

print_help() {
    cat <<EOF
Usage: $(basename "$0") [options]

Configures, builds, and runs pc_wslam, then launches the python presenter
on the exported map once the binary exits successfully.

Options:
  -h, --help        Show this help and exit.
  -gui              Enable the Pangolin visualization GUI.
  awaiter           Enable verbose awaiter logging (LOG_AWAITER_CALLS=ON).
  --max-iters=N     Stop after N top-level pipeline executions (0 = unlimited).
  --map-out=PATH    Path the final factor-graph map is written to. Must end
                    in .ply; a sibling .json metadata file is derived
                    automatically. Defaults to /tmp/uniwslam_map.ply when
                    omitted, since the presenter needs a map to load.
EOF
}

mkdir -p build

LOG_AWAITER=OFF
RUN_ARGS=()
MAP_OUT=""

for arg in "$@"; do
    case "$arg" in
        -h|--help)      print_help; exit 0 ;;
        awaiter)        LOG_AWAITER=ON ;;
        -gui)           RUN_ARGS+=("-gui") ;;
        --max-iters=*)  RUN_ARGS+=("$arg") ;;
        --map-out=*)    MAP_OUT="${arg#--map-out=}" ;;
        *)              echo "error: unknown argument '$arg'" >&2
                        print_help >&2
                        exit 1 ;;
    esac
done

# The presenter needs the produced map; pick a default path if the caller
# did not name one. We always append the flag to RUN_ARGS so the binary
# sees a consistent value.
if [[ -z "$MAP_OUT" ]]; then
    MAP_OUT="/tmp/uniwslam_map.ply"
    echo "No --map-out= provided; defaulting to $MAP_OUT"
fi
RUN_ARGS+=("--map-out=$MAP_OUT")

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLOG_AWAITER_CALLS=$LOG_AWAITER

cmake --build build

echo "BUILD FOR PROJECT pc_wslam SUCCESSFUL!!!"

export WSLAM_SHADER_SRC_DIR="$PWD"

source .env
./build/pc_wslam "${RUN_ARGS[@]}"

# Reaching this point means pc_wslam exited 0 (set -e bails earlier
# otherwise). Hand off to the presenter on the freshly exported map.
echo "Launching presenter on $MAP_OUT"

if [[ -x "presentor/.venv/bin/presenter" ]]; then
    exec presentor/.venv/bin/presenter "$MAP_OUT"
elif command -v presenter >/dev/null 2>&1; then
    exec presenter "$MAP_OUT"
else
    echo "error: presenter not installed. Either:" >&2
    echo "  - install it system-wide: pip install -e presentor/" >&2
    echo "  - or create a venv first: cd presentor && python -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi
