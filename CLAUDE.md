# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

Dawn, Pangolin and GTSAM must all be pre-built and installed into
`vendor/dawn/install/`, `vendor/pangolin/install/` and `vendor/gtsam/install/`
before CMake will succeed — they are not fetched by CMake. The repo ships
helper scripts `build_dawn.sh`, `build_pangolin.sh` and `build_gtsam.sh` for
this. **Eigen is pinned to 3.3.7** in `cmake/configureDependencies.cmake`
because GTSAM 4.2.1 statically asserts `EIGEN_MAJOR_VERSION == 3` at consumer
compile time.

```bash
# Configure, build, and run (sets WSLAM_SHADER_SRC_DIR automatically)
./run.sh

# Run and then launch the python presenter on the exported map
./present_and_run.sh

# Either script accepts the same flags
./run.sh -gui --max-iters=50 --map-out=/tmp/uniwslam_map.ply
./run.sh --help                       # full option list

# awaiter:    verbose logging of GPU future add/complete events
# -gui:       enable the Pangolin visualisation pass
# --max-iters=N: stop after N pipeline iterations (0 = unlimited)
# --map-out=PATH.ply: dump the final factor-graph map to disk
```

Or manually:
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
source .env
WSLAM_SHADER_SRC_DIR="$PWD/resources" ./build/pc_wslam
```

The `.env` file sets required environment variables:
- `WSLAM_SHADER_SRC_DIR` — directory where `.wgsl` shaders are loaded from at runtime
- `EUROC_DIR` — path to EuRoC MAV dataset (`mav0/` root)
- `TUM_FR1_RGBD_DIR` — path to TUM RGB-D dataset (provider currently commented out)

Debug builds enable ASan + UBSan by default (`ENABLE_SANITIZERS=TRUE`).

## Lint

```bash
# clang-tidy (config in .clang-tidy — excludes vendor/)
clang-tidy -p build <file.cpp>

# clang-format (config in .clang-format)
clang-format -i <file.cpp>
```

There is no automated test suite. `wslam/brief_tests.inc` is static BRIEF descriptor data, not a test file.

## Architecture

### Compute engine (`compute/`)

The pipeline is `Compute` → `Stage` → `Pass`:

- **`Compute`** owns a list of `Stage`s and a shared `AnyBag` storage. `execute()` runs every stage in order each frame. Returning the sentinel string `kComputeStopExecution` from a stage halts the loop.
- **`Stage`** is a named sequence of `Pass`es. Returning `kStageStopExecution` stops that stage and bubbles up.
- **`Pass`** is the leaf unit. `GPUPass` wraps a WebGPU compute pipeline with bind groups. `CustomPass` wraps an arbitrary CPU lambda — used for buffer clearing and CPU-side algorithm steps.

### GPU memory model (`compute/gpu.hpp`)

`GPU` pre-allocates fixed slabs at startup:

| Buffer | Size | Purpose |
|--------|------|---------|
| `StorageA` / `StorageB` | 500 MB each | Intermediate GPU compute data |
| `SharedStorage` | 64 MB | Cross-pass persistent data |
| `Input` | 64 MB | CPU→GPU uploads |
| `Output` | 64 MB | GPU→CPU readback |
| `Uniform` | 32 KB | Per-dispatch constants |

Sub-regions are allocated via `assignBuffersAndOffsets()` and returned as `BufferBinding` objects. `BufferBinding` is RAII — its destructor calls `GPU::unAssignBuffer()` to mark the region free.

`clearBuffersAndOffsets()` is called each frame by the `[Clear bindings]` stage (the first stage in the pipeline). It zeros the GPU memory of every region that is currently in use (`!is_free`) and not marked retained (`!is_retained`). This resets transient per-frame data (e.g. the feature array's atomic count) without touching constant data written once at `initialize()` time. Bindings that hold constant data (BRIEF test vectors, LOD index values) must be created with `is_retained = true`; all others default to `false` and are zeroed each frame.

### Cross-stage data exchange (`base/anybag.hpp`)

`AnyBag` is a string-keyed, type-erased store (`std::unordered_map<string, Any>`). Stages write results into `Compute::storage_` (accessible via `Stage::storage_`) and downstream stages read them by key. Keys are defined in `wslam::ResourceIdentifier` in `common.hpp` — always use those constants, never raw strings.

### Shared model types (`wslam/models.hpp`)

Cross-pass payload types live in `wslam/models.hpp`: `Feature`, `FeatureSet`, `MatchResult`, `RansacResult`, `TriangulationResult`, plus the factor-graph types `PoseId`, `LandmarkId`, `LandmarkObservation`, `MapDelta`, `KeyframePose`, `LandmarkEstimate`, `MapStats`, `MapSnapshot`. New cross-pass types belong here unless they pull in heavy headers (GTSAM types live in `wslam/mapping_state.hpp` instead, behind a pImpl-style boundary).

### SLAM pipeline (`wslam/`)

`CreateWslamPipeline()` in `wslam.hpp` assembles the pipeline. It accepts a `WslamConfig` (defined in `common.hpp`) that controls `enable_gui`, `max_iterations`, and `map_out_path`. It returns a `WslamPipelineHandles` struct holding `shared_ptr<MappingState>` — the caller must keep this alive for the duration of the run because the mapping stage's passes reference it.

0. **Clear bindings stage** — runs first every frame; calls `clearBuffersAndOffsets()` to zero non-retained GPU regions before new work is dispatched.
1. **Feature detect stage** (`CreateFeatureDetectStage`):
   - `SensorLoader` pass — reads next frame from the `std::generator` data provider, uploads to GPU input buffer
   - `FillPyramid` pass — builds a 6-level LoD image pyramid on GPU (scale 1.2× per level, hardcoded for 752×480)
   - `DetectCorners` + `CullCorners` passes — GPU FAST-style corner detection and NMS
   - `GenerateFeatures` pass — computes ORB-style BRIEF descriptors on GPU (bit patterns from `brief_tests.inc`)
   - `LoadFeaturesCPU` pass — reads features back to CPU via `Awaiter`

2. **Pose estimate stage** (`CreatePoseEstimateCPUStage`):
   - `MatchFeaturesCPU` pass — BRIEF descriptor matching on CPU with mutual cross-check, Lowe ratio test, and a 10%-frame spatial gate applied *before* the Hamming distance (so out-of-window candidates do not pollute the ratio test).
   - `RansacCPU` pass — normalised 8-point fundamental-matrix RANSAC with a Sampson-distance inlier test.
   - `TriangulateCPU` pass — undistorts the RANSAC inliers, re-fits the essential matrix on calibrated rays, decomposes into four (R, t) candidates, picks the one with the most cheirality passes, then runs linear DLT triangulation and a reprojection-error filter.

3. **Mapping stage** (`CreateMappingStage` in `wslam/mapping.hpp`) — incremental factor-graph SLAM via GTSAM iSAM2. State (the iSAM2 instance, the cached `Cal3DS2`, pose / landmark counters, the `flat_map<Feature, LandmarkId>` active-track map) lives in `MappingState`, injected by reference into all three passes:
   - `KeyframeGatePass` — gates on min-landmarks and min-rotation; allocates a new `PoseId`; chains the world pose off the previous keyframe's `latest_values` estimate; associates landmarks against the previous keyframe's `feat_curr → LandmarkId` map; for *new* landmarks introduced after the first keyframe, emits observations at **both** the previous and current pose (the prev keyframe is already in the graph, so a new landmark immediately has two views — keeping the linear system well posed).
   - `FactorBuilderPass` — constructs the per-frame `NonlinearFactorGraph` and `Values` delta: `PriorFactor<Pose3>` (gauge) on the first keyframe, `BetweenFactor<Pose3>` (odometry, with the inverse-transpose direction conversion that GTSAM's between convention requires), Huber-robust `GenericProjectionFactor<Pose3, Point3, Cal3DS2>` per observation, plus per-landmark anchor priors (tight on the first keyframe, soft on later ones — needed to regularise low-parallax cases).
   - `Isam2UpdatePass` — `ISAM2::update(new_factors, new_values)`, optional extra updates, extracts `calculateEstimate()` into `MappingState::latest_values`, and publishes a GTSAM-free `MapSnapshot` to AnyBag. Catches `IndeterminantLinearSystemException` with the offending key formatted in the error message.

4. **Map export** (`wslam/export.hpp`, called from `main.cpp` *after* the pipeline loop exits, not as a pass): writes `MapSnapshot` and `CamSensorParams` to `<stem>.ply` (landmark point cloud, ASCII PLY with `x`, `y`, `z`, `landmark_id`) and `<stem>.json` (keyframes, intrinsics, stats, schema). Triggered only when `config.map_out_path` is non-empty.

### Python presenter (`presentor/`)

`presentor/` is a small PEP-621 `pyproject.toml` package providing the `presenter` CLI (`python -m presenter` or the `presenter` entry point). Dependencies: `rerun-sdk` (visualisation), `plyfile` (PLY loader), `numpy`. The viewer logs the landmark cloud, keyframe frustums, and the trajectory polyline into rerun.

`viewer.py` is split into `init` / `log_landmarks` / `log_camera` / `log_trajectory` helpers so the same pieces can drive an incremental loader later (`rr.set_time_sequence(...)` per arriving keyframe). The schema the loader expects is documented in `wslam/export.cpp` (`format_version = 1`).

### Data providers (`data/`)

Providers implement `ProviderBase<N>` and expose a `std::generator<std::expected<Reading<N>, std::string>> getReadings()` coroutine. `AdaptProvider<Requested, Available>` adapts an N-camera provider to an M-camera consumer. Currently only `EurocProvider` is active; `TumProvider` is compiled but not wired in `CMakeLists.txt`.

### Error handling convention

- Functions that can fail return `std::optional<std::string>` — `std::nullopt` means success, a string is the error message.
- Functions that return values use `std::expected<T, std::string>`.
- WGSL shaders are loaded and validated at `initialize()` time, not at dispatch time.
- Per-pass logs use a `LOG_ID` macro (`#define LOG_ID "[Pass Name]"`) at the top of each `.cpp`; all spdlog calls prefix with it for grep-friendly timelines.
