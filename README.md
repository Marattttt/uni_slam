# WebGPU SLAM - a cross-platform performant solution for navigation and spatial reasoning

The project is based on principles established by ORB-SLAM3 and serves as an example of using WebGPU 
to parallelize computations in a cross platform way without using the Nvidia CUDA platform

## Modernization (in progress)

The codebase is being modernized incrementally. Current checklist:

- [x] Divide the single project into multiple CMake targets.
- [ ] Add a testing API into the compute library.
- [ ] Implement tests for the wslam passes.

## Architecture

The core solution is a C++23 CMake project split into a small set of libraries that a
thin executable links together:

- base: a header-only interface library of shared utilities.
- compute: organizes both WebGPU and custom computations into a single pipeline API.
- data: provides an interface for loading data and an implementation for EuRoC.
- wslam: provides a SLAM pipeline based on ORB-SLAM3.

The compute, data and wslam libraries keep a public interface separate from their
private implementation, so each depends only on the public surface of the others.
Shared warning and sanitizer settings are applied uniformly across the project's own
targets.

The presenter directory contains a small python wrapper around the Rerun SLAM
visualizer which is used to inspect generated maps.

## How the core solution is made

C++23 with GCC 15 and CMake. Each library declares only the dependencies it needs
and exposes them to its consumers through its public interface, so the executable and
the other libraries pull in exactly what they use.

**Vendored heavy dependencies**
- GTSAM: factor-graph smoothing framework. A major architecture change from ORB-SLAM3 made in order to allow switching to lag solvers
- Google Dawn: implementation of the WebGPU spec used in Google Chrome
- Pangolin: OpenGL frontend for debug purposes

Other dependencies are brought in through FetchContent:

- spdlog: logging. Header-only, built with `std::format` (`SPDLOG_USE_STD_FORMAT`).
- Eigen: linear algebra. Pinned to 3.4.0 because GTSAM 4.3a1 bundles Eigen 3.4.0 and statically asserts the WORLD/MAJOR/MINOR version matches the consumer's at compile time.
- stb: image loading.
- ctre: compile-time regular expressions.
- nlohmann_json: JSON output for the exported map.
- yaml-cpp: dataset/config parsing.
- fast-cpp-csv-parser: reads EuRoC ground-truth and IMU CSVs.

## Build & Run

Dawn, Pangolin and GTSAM must all be pre-built and installed into
`vendor/dawn/install/`, `vendor/pangolin/install/` and `vendor/gtsam/install/`
before CMake will succeed — they are not fetched by CMake. The repo ships
helper scripts `build_dawn.sh`, `build_pangolin.sh` and `build_gtsam.sh` for
this.

```bash
# Configure (Debug), build, and run (sources .env, sets WSLAM_SHADER_SRC_DIR)
./run.sh

# Same but Release build, then launch the python presenter on the exported map
./present_and_run.sh

# Either script accepts the same flags
./run.sh -gui --max-iters=50 --map-out=/tmp/uniwslam_map.ply
./run.sh --help                       # full option list
```

Flags:
- `-gui` — enable the Pangolin visualisation pass (matches, RANSAC inliers, projected landmarks).
- `--max-iters=N` — stop after N pipeline iterations (0 = unlimited).
- `--map-out=PATH.ply` — dump the final factor-graph map to disk (`.ply` cloud + `.json` metadata).
- `awaiter` — verbose logging of GPU future add/complete events.
- `Release | Debug` - build configuration can be set. Note: Release resolves to Release with debug info

Manual configure/build/run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
WSLAM_SHADER_SRC_DIR="$PWD/resources" \
    EUROC_DIR="<path to euroc dataset's mav0 directory>" \
    ./build/pc_wslam
```

## Pipeline

Each frame runs a fixed sequence of stages built by `CreateWslamPipeline()`:

1. **Feature detect** (GPU) — LoD image pyramid, FAST-style corner detection + NMS, ORB-style BRIEF descriptors.
2. **Pose estimate** (CPU) — BRIEF matching against the previous keyframe, 8-point fundamental-matrix RANSAC, essential-matrix decomposition and DLT triangulation.
3. **Mapping** — incremental visual-inertial SLAM via GTSAM iSAM2 with smart projection factors and `CombinedImuFactor` between keyframes, optimised asynchronously on a worker thread.
4. **Map export** — writes the landmark cloud and keyframe trajectory to `<stem>.ply` / `<stem>.json`.

The compute engine (`compute/`) batches consecutive GPU passes into a single
WebGPU command submission per stage. See `CLAUDE.md` for the full architecture.

## Presenter

`presentor/` is a small PEP-621 Python package exposing the `presenter` CLI. It
loads an exported map and logs the landmark cloud, keyframe frustums and the
trajectory polyline into [Rerun](https://rerun.io).

```bash
python -m presenter <map>.ply        # or use present_and_run.sh
```

## Benchmarks

`benchmarks/` evaluates accuracy against ORB-SLAM3 and EuRoC ground truth using
the [evo](https://github.com/MichaelGrupp/evo) toolset. ORB-SLAM3 is vendored as
a git subtree. See `benchmarks/ACCURACY_ANALYSIS.md` for current numbers and
`benchmarks/HOWTO_REEVALUATE.md` for the re-evaluation checklist.

