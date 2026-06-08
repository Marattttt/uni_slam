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
# Configure (Debug), build, and run (sources .env, sets WSLAM_SHADER_SRC_DIR)
./run.sh

# Same but Release build, then launch the python presenter on the exported map
./present_and_run.sh

# Either script accepts the same flags
./run.sh -gui --max-iters=50 --map-out=/tmp/uniwslam_map.ply
./run.sh --help                       # full option list
```

Flags:
- `awaiter` — verbose logging of GPU future add/complete events (`-DLOG_AWAITER_CALLS=ON`).
- `-gui` — enable the Pangolin visualisation pass.
- `--max-iters=N` — stop after N pipeline iterations (0 = unlimited).
- `--map-out=PATH.ply` — dump the final factor-graph map to disk.

Manual configure/build:
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
source .env
WSLAM_SHADER_SRC_DIR="$PWD/resources" ./build/pc_wslam
```

The `.env` file is only needed at **runtime** — configuring and building the
project (`cmake -S . -B build ...`, `cmake --build build`) does not read any of
these variables, so there is no need to `source .env` before a build. Source it
only when executing `pc_wslam` (or the helper run scripts, which do it for you).
It sets:
- `WSLAM_SHADER_SRC_DIR` — directory where `.wgsl` shaders are loaded from at runtime.
- `EUROC_DIR` — path to EuRoC MAV dataset (`mav0/` root).
- `TUM_FR1_RGBD_DIR` — path to TUM RGB-D dataset (provider currently not wired in `CMakeLists.txt`).

Debug builds enable ASan + UBSan by default (`ENABLE_SANITIZERS=TRUE`).

## Lint

```bash
clang-tidy -p build <file.cpp>     # config in .clang-tidy — excludes vendor/
clang-format -i <file.cpp>         # config in .clang-format
```

There is no automated test suite. `wslam/brief_tests.inc` is static BRIEF
descriptor data, not a test file.

## Architecture

### Compute engine (`compute/`)

Hierarchy: `Compute` → `Stage` → (`Pass` | `GPUPass`).

- **`Compute`** owns a list of `Stage`s and a shared `AnyBag` storage. `execute()` runs every stage in order each frame. Returning the sentinel string `kComputeStopExecution` from a stage halts the loop.
- **`Stage`** is a named, ordered sequence of passes stored as `std::variant<unique_ptr<Pass>, unique_ptr<GPUPass>>`. Returning `kStageStopExecution` stops that stage and bubbles up.
- **`Pass`** is a pure-CPU leaf. Implements `initialize()` and `execute()`.
- **`GPUPass`** is a sibling class (NOT a subclass of `Pass`) for WebGPU compute work. Implements `initialize()` and `prepareExecute(const wgpu::CommandEncoder&)` — the encoder is owned by the stage, not the pass.
- **`CustomPass`** wraps an arbitrary CPU lambda; used for buffer clearing and one-off algorithm steps.

GPU dispatch is **batched per stage**: `Stage::execute()` walks its passes, collects every consecutive `GPUPass` into a single `wgpu::CommandEncoder`, then issues one `queue.Submit` and one `OnSubmittedWorkDone` await for the whole batch. A CPU `Pass` between two GPU passes flushes the in-progress batch first. This is why GPU passes no longer call `queue.Submit` themselves — they only record commands into the supplied encoder and return an error string on failure.

The `Awaiter` (in `compute/awaiter.hpp`) splits async work into two methods that must not be mixed in one chain when `executeAll` uses a non-zero timeout: `runChecked(callback, label)` wraps a synchronous API call in `PushErrorScope`/`PopErrorScope` (producing `WaitListEvent` futures) for validation-prone init code, while `addFuture(factory, label)` records a `wgpu::Future` returned by the factory (queue-serial for `OnSubmittedWorkDone`/`MapAsync`, no error scopes). Dawn rejects a `WaitAny` whose future set mixes those two sources (`EventManager.cpp` "Mixed source waits with timeouts are not currently supported"). For the common "submit a command buffer and wait for it" pattern, call `GPU::submitAndWait(commands, label, timeout)` instead of assembling it from an `Awaiter` by hand.

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

`AnyBag` is a string-keyed, type-erased store (`std::unordered_map<string, Any>`). Stages write results into `Compute::storage_` (accessible via `Stage::storage_`) and downstream stages read them by key. Keys are defined in `wslam::ResourceIdentifier` in `common.hpp` — always use those constants, never raw strings. Note the constexpr key strings must fit libstdc++'s 15-char SSO buffer. `take<T>(key)` moves the value out and erases the entry — the consume-once primitive used for cross-stage markers (e.g. `FeatureReferenceAdvanceName`).

### Shared model types (`wslam/models.hpp`)

Cross-pass payload types live in `wslam/models.hpp`: `Feature`, `FeatureSet`, `MatchResult`, `RansacResult`, `TriangulationResult`, plus the factor-graph types `PoseId`, `LandmarkId`, `LandmarkObservation`, `MapDelta`, `KeyframePose`, `LandmarkEstimate`, `MapStats`, `MapSnapshot`. New cross-pass types belong here unless they pull in heavy headers (GTSAM types live in `wslam/mapping_state.hpp` instead, behind a pImpl-style boundary).

### Pass constructor convention

CPU passes (`Pass` subclasses) must NOT take a `std::shared_ptr<compute::GPU>` in their constructor unless they actually call GPU APIs. The handful that do read back from the GPU (e.g. `LoadDataCPUPass` calls `gpu_->readTexture` / `readBuffer`, `VisualizeDataPass` uses the device for Pangolin texture upload) store `gpu_` as a private field — they do not inherit it. This was tidied across the codebase in the `refactor/bundle-gpu-passes-execution` work; preserve the convention when adding new passes.

GPU passes (`GPUPass` subclasses) inherit `gpu_` from the base class.

### SLAM pipeline (`wslam/`)

`CreateWslamPipeline()` in `wslam.hpp` assembles the pipeline from a `WslamConfig` (defined in `common.hpp`) that controls `enable_gui`, `max_iterations`, and `map_out_path`. It returns a `WslamPipelineHandles` struct holding `shared_ptr<MappingState>` and a `flush_async` callable — the caller must keep this alive for the duration of the run because the mapping stage's passes reference the state, and must call `flush_async()` after the loop exits but before consumers (e.g. `ExportMap`) read `MapSnapshotName`.

0. **Clear bindings stage** — a single `CustomPass` runs first every frame; calls `clearBuffersAndOffsets()` to zero non-retained GPU regions before new work is dispatched.

1. **Feature detect stage** (`CreateFeatureDetectStage`):
   - `SensorLoaderPass` (CPU) — pulls the next frame from the `std::generator` data provider, publishes the FrameBW to AnyBag, accumulates IMU samples into a vector under `GetImuVecName()`, and publishes the current frame's timestamp under `FrameTimestampNsName`. `kInitialFramesToSkip` is **0**: the stationary lead-in of the dataset is required — gravity and gyro bias are estimated from it and the first keyframe's zero-velocity prior assumes it (a previous 200-frame skip jumped past V1_01_easy's takeoff and forced a mid-flight gravity estimate, `benchmarks/ACCURACY_ANALYSIS.md` R3).
   - `FillPyramidPass` (GPU) — builds a 6-level LoD image pyramid (scale 1.2× per level, hardcoded for 752×480).
   - `PassDetectCorners` + `CullCornersPass` (GPU) — FAST-style corner detection and NMS.
   - `GenerateFeaturesPass` (GPU) — computes ORB-style BRIEF descriptors (bit patterns from `brief_tests.inc`).
   - `LoadDataCPUPass` runs in the *next* stage (see below); the feature detect stage ends after `GenerateFeaturesPass` so all GPU work batches into a single submission.

2. **Pose estimate stage** (`CreatePoseEstimateCPUStage`):
   - `LoadDataCPUPass` — reads features (and, GUI only, LoD textures) back to CPU via `Awaiter`. Maintains the **reference feature set** (`FeatureSet(1)`): pinned to the last accepted keyframe's features and advanced only when the keyframe gate publishes the consume-once `FeatureReferenceAdvanceName` marker (every frame pre-origin, acceptance only after). All downstream matching/geometry therefore spans *previous keyframe → current frame* — exactly the factor-graph edge.
   - `MatchFeaturesCPU` — BRIEF descriptor matching (reference ↔ current) on CPU with mutual cross-check, Lowe ratio test at **0.85** (looser than textbook 0.75: V101's repetitive texture makes tight ratios reject true matches quasi-randomly, starving re-observation), and a 10%-frame spatial gate applied *before* the Hamming distance. The per-LOD candidate pool is the same LOD ± 1 (same-LOD inclusion is essential; an earlier version searched only neighbouring LODs and capped the match yield at the few % of scale-flickering features). Missing reference (first frame) stops only the stage (`kStageStopExecution`) so the mapping stage still bootstraps.
   - `RansacCPU` — normalised 8-point fundamental-matrix RANSAC with a Sampson-distance inlier test.
   - `TriangulateCPU` — undistorts the RANSAC inliers, re-fits the essential matrix on calibrated rays, decomposes into four (R, t) candidates, picks the one with the most cheirality passes, then runs linear DLT triangulation and a reprojection-error filter.
   - `VisualizeDataPass` (when `enable_gui`) — Pangolin GUI for matches, RANSAC inliers, and projected landmarks.

3. **Mapping stage** (`CreateMappingStage` in `wslam/mapping.hpp`) — incremental visual-inertial SLAM via GTSAM iSAM2 with **smart projection factors**. State (the iSAM2 instance, the cached `Cal3_S2`, pose / landmark counters, the `flat_map<Feature, LandmarkId>` active-track map, IMU bias / velocity propagation, gravity estimate) lives in `MappingState`, injected by reference into all four passes. The frame-vs-keyframe geometry conflation that originally capped accuracy (R1 in `benchmarks/ACCURACY_ANALYSIS.md`) is fixed by the reference-feature-set mechanism above; evaluation history and remaining gaps are tracked in that document. Stage order:
   - `Isam2DrainPass` — runs **first** so the keyframe gate and factor builder see up-to-date pose / smart-factor estimates from the previous iSAM round. Without this, the factor builder would read stale `smart_factor_indices` and schedule removal of factor indices iSAM had already replaced.
   - `KeyframeGatePass` — gates on min-landmarks, min-rotation OR min-parallax, and a min count of *new* landmarks. Allocates a new `PoseId`; chains the world pose off the previous keyframe's `predicted_values` (NOT `latest_values` — the previous iSAM update may still be in-flight on the worker). Associates **all RANSAC inlier pairs** (not just triangulated landmarks — smart factors need no pre-validated 3D point, and associating on the triangulated subset capped re-observation at chance level) against the previous keyframe's `feat_curr → LandmarkId` map. Bootstraps the first keyframe at **motion onset** (small `bootstrap_parallax_px` vs the per-frame-advancing reference), then initialises gravity *and gyro bias* from the quietest stationary window found in the IMU buffer (min `std(|accel|) + mean gyro deviation` score — absolute thresholds fail: EuRoC's gyro bias is ~0.08 rad/s and a parked MAV vibrates).
   - `FactorBuilderPass` — constructs the per-frame `NonlinearFactorGraph` and `Values` delta: `PriorFactor<Pose3>` (gauge) on the first keyframe, `BetweenFactor<Pose3>` (vision-only odometry, with the inverse-transpose direction conversion that GTSAM's between convention requires), `CombinedImuFactor` between consecutive keyframes, priors on initial velocity and IMU bias, and one `SmartProjectionPoseFactor<Cal3_S2>` per landmark. Smart factors marginalise the 3D point out of the optimisation, sidestepping iSAM2's depth-axis singularities on low-parallax landmarks. Observations are **pre-undistorted upstream** (in the gate pass) so the smart factor sees a clean pinhole `Cal3_S2`. On re-observations, the existing smart factor is mutated (`add()`) and re-pushed to iSAM2 via the remove-and-readd pattern, with the old factor index drained out of `smart_factor_indices`.
   - `Isam2UpdatePass` — submits `(new_factors, new_values, remove_indices)` to an `Isam2Worker` running on a dedicated thread. The pass is asynchronous: each `execute()` drains the *previous* frame's pending result (blocking only if iSAM hasn't caught up), applies it to `MappingState::latest_values`, publishes a GTSAM-free `MapSnapshot` under `MapSnapshotName`, then submits the current frame's work. Catches `IndeterminantLinearSystemException` with the offending key formatted in the error message. Drained explicitly by `Isam2DrainPass` at the top of the next frame's stage; `flush()` is called once after the main loop exits to absorb the final keyframe's optimisation.

4. **Map export** (`wslam/export.hpp`, called from `main.cpp` *after* the pipeline loop exits, not as a pass): writes `MapSnapshot` and `CamSensorParams` to `<stem>.ply` (landmark point cloud, ASCII PLY with `x`, `y`, `z`, `landmark_id`) and `<stem>.json` (keyframes, intrinsics, stats, schema). Triggered only when `config.map_out_path` is non-empty.

### Python presenter (`presentor/`)

`presentor/` is a small PEP-621 `pyproject.toml` package providing the `presenter` CLI (`python -m presenter` or the `presenter` entry point). Dependencies: `rerun-sdk` (visualisation), `plyfile` (PLY loader), `numpy`. The viewer logs the landmark cloud, keyframe frustums, and the trajectory polyline into rerun.

`viewer.py` is split into `init` / `log_landmarks` / `log_camera` / `log_trajectory` helpers so the same pieces can drive an incremental loader later (`rr.set_time_sequence(...)` per arriving keyframe). The schema the loader expects is documented in `wslam/export.cpp` (`format_version = 1`).

### Benchmarks (`benchmarks/`)

Accuracy evaluation against ORB-SLAM3 and EuRoC ground truth using the
[evo](https://github.com/MichaelGrupp/evo) toolset. `benchmarks/ORB_SLAM3/` is
a **git subtree** (squashed) of upstream `UZ-SLAMLab/ORB_SLAM3` `master` with
two local patches (C++17 bump; `LoopClosing.h` `mnFullBAIdx` bool→int) — when
modifying it, remember the changes are tracked in this repo's history.

- `benchmarks/build_orbslam3.sh` — builds ORB-SLAM3 + its Thirdparty against
  system OpenCV and the project's `vendor/pangolin/install` (needs
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` under CMake ≥ 4).
- `benchmarks/run_orbslam3_euroc.sh` — runs mono-inertial on the `.env`
  EuRoC sequence; trajectories land in `benchmarks/results/`. The binary
  segfaults at shutdown *after* saving (known upstream viewer/GL teardown
  race) — the script tolerates it and validates the output files instead.
- `benchmarks/compare.py` (run with `benchmarks/.venv/bin/python`; venv has
  `evo` + `numpy`) — converts wslam `map.json`, ORB-SLAM3 txt (ns
  timestamps), and EuRoC GT csv to TUM format and runs `evo_ape`/`evo_rpe`.
- `benchmarks/ACCURACY_ANALYSIS.md` — current accuracy numbers (wslam 1.62 m
  vs ORB-SLAM3 0.036 m APE on V101), evidence-backed root causes, and the
  ranked improvement plan. Read this before touching mapping-stage accuracy.
- `benchmarks/HOWTO_REEVALUATE.md` — the re-evaluation checklist: run with a
  per-experiment `--out-dir`, check Umeyama scale → RPE → APE in that order,
  and verify root-cause fixes with the debug-log greps (re-observation rate,
  `metric_step`, gravity line).

### Data providers (`data/`)

Providers implement `ProviderBase<N>` and expose a `std::generator<std::expected<Reading<N>, std::string>> getReadings()` coroutine. `AdaptProvider<Requested, Available>` adapts an N-camera provider to an M-camera consumer. Currently only `EurocProvider` is active; `TumProvider` is compiled but not wired in `CMakeLists.txt`.

### Error handling convention

- Functions that can fail return `std::optional<std::string>` — `std::nullopt` means success, a string is the error message.
- Functions that return values use `std::expected<T, std::string>`.
- WGSL shaders are loaded and validated at `initialize()` time, not at dispatch time.
- Per-pass logs use a `LOG_ID` macro (`#define LOG_ID "[Pass Name]"`) at the top of each `.cpp`; all spdlog calls prefix with it for grep-friendly timelines.
