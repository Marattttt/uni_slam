# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

**This machine has no discrete GPU.** Pass `--device=cpu` for most runs (e.g.
`./run.sh -v2 --device=cpu ...`); with the default `--device=gpu` the project
picks the integrated GPU, which is slow.

ONNX Runtime, Pangolin and GTSAM must all be pre-built and installed into
`vendor/onnxruntime/install/`, `vendor/pangolin/install/` and
`vendor/gtsam/install/` before CMake will succeed — they are not fetched by
CMake. The repo ships helper scripts `scripts/build_onnxruntime.sh`,
`scripts/build_pangolin.sh` and `scripts/build_gtsam.sh` for this (run them from the
repo root: they resolve `vendor/` from `$PWD`). **There is no separate Dawn
build**: `scripts/build_onnxruntime.sh` configures ORT with
`onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON`, so the WebGPU execution provider
build emits an installable `libwebgpu_dawn.so`, and both
`onnxruntime::onnxruntime` and `dawn::webgpu_dawn` come out of that one install
prefix. Building Dawn separately as well would put two independent WebGPU
implementations in one process. Two consequences worth knowing: ORT declares
Dawn `EXCLUDE_FROM_ALL` *and* its FetchContent wrapper forces
`CMAKE_SKIP_INSTALL_RULES`, so Dawn is installed by a second, explicit
`cmake --install` on its own build directory and by a local patch to that
wrapper (see `vendor/onnxruntime/cmake/external/onnxruntime_external_deps.cmake`,
marked `LOCAL (uni_slam)`); and Dawn is now a **shared** library where it used
to be a static archive. The same script also builds **SwiftShader**, Dawn's CPU
adapter used by `--device=cpu`: Dawn's dependency fetcher skips it, so the script
checks it out into `vendor/onnxruntime/build/swiftshader` at the commit pinned in
Dawn's `DEPS` (it fails if Dawn moves), configures Dawn with
`DAWN_ENABLE_SWIFTSHADER=ON`, and installs `libvk_swiftshader.so` next to
`libwebgpu_dawn.so`, which is where Dawn looks for it at run time. **FAISS** (loop closure) is the one FetchContent dependency with a native build:
v1.15.1 by tarball hash, CPU only, linking the system BLAS/LAPACK and OpenMP.
**Eigen is pinned to 3.4.0** in `cmake/configureDependencies.cmake`
because GTSAM 4.3a1 bundles Eigen 3.4.0 and statically asserts the Eigen
WORLD/MAJOR/MINOR version matches the consumer's at compile time.

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
- `-v2` — run the v2 pipeline (ONNX feature detect + pose estimate + fixed-lag mapping) instead of the default one.
- `--device=gpu|cpu` — where compute runs (`compute::ComputeDevice`, `WslamConfig::compute_device`;
  default `gpu`). `cpu` forces Dawn onto its SwiftShader CPU adapter (`forceFallbackAdapter`,
  which discards every other adapter) and, with `-v2`, runs the models on ONNX Runtime's CPU EP
  (`OrtCpuFeatureDetectPass`). `-v2` with `gpu` currently fails at startup: only CPU exports of
  the split models exist (`v2::ValidateModelFiles`, a `TEMPORARY` block in `feature_detect.cpp`;
  the WebGPU path, `OrtFeatureDetectPass`, is ready for the WebGPU exports). `gpu` requests a `HighPerformance` adapter: Dawn searches
  SwiftShader before the system drivers and, without a power preference, ranks all adapter
  types equally, so the default request lands on SwiftShader. `GPU::initAdapter` rejects an
  adapter whose type doesn't match the requested device either way.
- `-no-mapping` — v2 only: drop the mapping stage; the model then pairs consecutive frames.
- `-no-loop-closure` — v2 with mapping: run without the loop closure thread
  (`WslamConfig::enable_loop_closure`, default on).
- `-no-frame-prefilter` — v2 with mapping: run the model on every frame instead of skipping
  the frames the IMU predicts the keyframe gate would reject
  (`WslamConfig::enable_frame_prefilter`, default on; off with `--keyframes`).
- `--record-frontend=PATH.msgpack` — v2 with mapping only: record the front-end output
  for replaying the mapping stage without the GPU (see `MappingReplayTest` below).
- `--max-iters=N` — stop after N pipeline iterations (0 = unlimited).
- `--map-out=PATH.ply` — dump the final factor-graph map to disk.
- `--keyframes=PATH.json` — v2 only: feed the pipeline only the frames of a keyframe
  schedule (`wslam/src/keyframe_schedule.hpp`), so every model run pairs two
  consecutive keyframes; all IMU is still read. `wslam/tests/data/v1_keyframes_V101.json`
  holds the keyframes v1 selected on V1_01_easy. The loop does not end by itself at
  the end of the stream (see Data providers below), so pair it with `--max-iters`.

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
- `WSLAM_ONNX_EXTRACTOR_MODEL`, `WSLAM_ONNX_MODEL` — the split RaCo-ALIKED-LightGlue models
  the v2 pipeline runs (both required with `-v2`; no default): the extractor
  (`resources/raco_aliked_lightglue_extractor_k512_boundary_cpu.onnx`, run once) and the
  cached model (`..._cached_k512_boundary_cpu.onnx`, every later frame; also the loop
  closure matcher). Exports of `LightGlue-ONNX/experiments/cpu_split_ranker/export.sh`
  (`--graph extractor` / `--graph cached --pad-matches`). CPU exports must have `cpu` in
  their file names.
- `WSLAM_ONNX_KEYPOINTS` — optional keypoints per image the models return; defaults to
  512 (`raco_lightglue::kDefaultKeypointCount`). Set it when testing another model
  variant; both models' shapes are checked against it when they load.
- `WSLAM_PREFILTER_SIGMAS` (default 2), `_GYRO_BIAS_SIGMA`, `_ACCEL_BIAS_SIGMA`,
  `_ATTITUDE_SIGMA`, `_VELOCITY_SIGMA`, `_DEPTH_SIGMA`, `_MAX_SKIPS`, `_STATIONARY_WINDOW`,
  `_STATIONARY_SCORE`, `_MAX_STATIONARY_SKIPS` (all `WSLAM_PREFILTER_*`) — override the
  frame prefilter's `FramePrefilterOptions` (`feature_detect_v2/frame_prefilter.hpp`).
- `WSLAM_VPR_MIXVPR_MODEL`, `WSLAM_VPR_MEGALOC_MODEL` — place recognition models of the
  v2 loop closure (required with loop closure on: MixVPR for `--device=cpu`, MegaLoc for
  `--device=gpu`). Exports of `~/dev/test/ort-export`: `mixvpr_320x320.onnx`,
  `megaloc_322x224_fp16.onnx` (or `_322x322_fp16`; the size is read from the model).
  `WSLAM_VPR_MODEL=megaloc|mixvpr` overrides the choice; `WSLAM_LC_TOP_N`,
  `WSLAM_LC_MIN_SIMILARITY`, `WSLAM_LC_MIN_KEYFRAME_GAP`, `WSLAM_LC_MIN_INLIERS`,
  `WSLAM_LC_NODE_MIN_TRANSLATION`, `WSLAM_LC_NODE_MIN_ROTATION`, `WSLAM_LC_THREADS`
  override `LoopClosureConfig` knobs (`map_v2/loop_closure/config.hpp`).
- `WSLAM_GRAVITY_MAGNITUDE` (9.81), `WSLAM_INIT_ZUPT_SCORE` (0.5),
  `WSLAM_INIT_VELOCITY_SIGMA` (0.05 m/s), `WSLAM_INIT_VELOCITY_SIGMA_PER_S` (0.2),
  `WSLAM_INIT_TILT_SIGMA_DEG` (3; 0 locks x1's tilt as before), `WSLAM_INIT_ACCEL_BIAS_SIGMA`
  (0.1) — the v2 IMU initialisation (`FilterKeyframeConfig`, `BuildFactorsPass::Opts`).
- `WSLAM_GLOBAL_BA` (1; 0 disables), `WSLAM_GBA_MAX_ITERS` (15), `WSLAM_GBA_MIN_OBS` (3) —
  the export-time global bundle adjustment after loop closure.
- `WSLAM_ORT_THREADS` — optional intra-op threads of the front end's ORT CPU session.
  By default ORT takes one per physical core; with loop closure on `--device=cpu` the
  pipeline gives it the physical cores minus `WSLAM_LC_THREADS` instead.
- `WSLAM_FRONTEND_RECORDING` — optional front-end recording for `MappingReplayTest`
  (kept outside the repo).
- `TUM_FR1_RGBD_DIR` — path to TUM RGB-D dataset (the TUM provider has been removed, so this variable is currently unused).

ASan + UBSan are opt-in via `-DENABLE_SANITIZERS=TRUE` (default off). When enabled
they are applied to the project's own targets through the shared `project_options`
interface target, independent of build type.

## Lint

```bash
clang-tidy -p build <file.cpp>     # config in .clang-tidy — excludes vendor/
clang-format -i <file.cpp>         # config in .clang-format
```

GoogleTest suites are built with `-DBUILD_TESTS=ON` (fetches GoogleTest):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build
./build/compute/tests/compute_tests
./build/wslam/tests/wslam_tests --gtest_filter=-*Dataset*   # no GPU / dataset needed
source .env && ./build/wslam/tests/wslam_tests --gtest_filter=*Dataset*
```

`wslam_tests` (`wslam/tests/`) reaches the library's internal headers by bare name and
reads test data from `wslam/tests/data/`. `PoseEstimateDatasetTest` runs v1 and v2 pose
estimation over the V1_01_easy keyframe schedule and scores both against EuRoC ground
truth (about 4 min in Debug, mostly ORT); it skips without `EUROC_DIR`,
`WSLAM_SHADER_SRC_DIR` and `WSLAM_ONNX_MODEL`, and `WSLAM_POSE_REPORT_OUT=path.md`
saves its report. `PoseEstimateDatasetTest.V2MatchScoreAblation` runs the model once and
re-scores the v2 geometry under every score-floor / weighting / refinement config
(`WSLAM_ABLATION_REPORT_OUT=path.md`). Results and analysis:
`benchmarks/POSE_ESTIMATE_V2_NOTE.md`.
`LocalSmootherTest`/`JobWorkerTest` cover the v2 mapping back-end machinery (no data).
`MotionPredictionTest`/`FramePrefilterPassTest` cover the frame prefilter and the IMU
initialisation helpers on synthetic IMU data (no data). `GlobalBaTest` runs the global
bundle adjustment on a synthetic circling run whose initial guess drifted in scale and yaw.
`LoopClosureModelIoTest`, `PlaceDatabaseTest`, `PnpVerifyTest` and `PoseGraphTest` cover
the loop closure pieces (no data); `Models/LoopClosureDatasetTest` checks each place
recognition model whose path is set on a ground-truth V1_01_easy revisit (MegaLoc on
the CPU EP takes ~3 min per image — it is a GPU model; `WSLAM_LC_PARITY_OUT=prefix`
dumps a descriptor for a Python parity check).
`MappingReplayTest` replays a front-end recording (`WSLAM_FRONTEND_RECORDING=path.msgpack`,
made by `pc_wslam -v2 --max-iters=N --record-frontend=PATH`; skips without it) through
the CPU RANSAC/triangulation passes and the v2 mapping stage, fails at the first
keyframe decision that differs from the recording (re-record after changing the gate)
and checks the final map against the recording run's; `WSLAM_REPLAY_MAP_OUT=map.ply`
exports the replayed map for `benchmarks/compare.py`. Most of its time is
`TriangulatePass`'s refinement, so it is much faster in an optimised build.
`MappingReplayTest.PrefilterPredictsTheGate` runs the prefilter's prediction on every
replayed frame (skipping nothing) and fails if it predicts small motion on a frame the gate
accepted; its `[Prefilter replay]` line reports the share of frames it would skip, and the
`WSLAM_PREFILTER_*` knobs apply.
`wslam/src/feature_detect/brief_tests.inc` is static BRIEF descriptor data, not a
test file.

## Architecture

### Project layout (CMake targets)

The build is split into a header-only interface library plus three compiled
libraries, linked by the thin `pc_wslam` executable (`main.cpp`):

- `base/` (`wslam::base`, INTERFACE) — shared header-only utilities (`anybag.hpp`,
  `assert.hpp`, `unique_any.hpp`, `job_worker.hpp`). Kept flat: `base/` itself is the include root, so
  consumers include e.g. `"anybag.hpp"`. Carries the `stdc++exp` link requirement of
  `WSLAM_ASSERT`'s `std::stacktrace` as an INTERFACE dependency, so consumers inherit it.
- `compute/` (`wslam::compute`) — public headers in `compute/include/compute/`, sources
  in `compute/src/`.
- `data/` (`wslam::data`) — public headers in `data/include/data/`, sources in `data/src/`.
- `wslam/` (`wslam::app`) — the public API is only `wslam/include/wslam/wslam.hpp` and
  `common.hpp`; everything else lives under `wslam/src/` (`feature_detect/`, `map/`,
  `pose_estimate_cpu/`, `viz/`, the v2 `feature_detect_v2/`, `pose_estimate_v2/`,
  `map_v2/`, plus `common.cpp`, `export.*`, `frontend_recording.*`, `models.hpp`,
  `mapping_state.hpp`).

Each compiled library exposes its public directory with
`target_include_directories(... PUBLIC include/)` and links dependencies with the right
PUBLIC/PRIVATE scope, so a consumer sees only the public surface. Internal `.cpp` that
include their own headers by bare name get a matching PRIVATE include dir. Shared
warning/sanitizer flags live on a single `project_options` INTERFACE target in the root
`CMakeLists.txt`.

### Compute engine (`compute/`)

Hierarchy: `Compute` → `Stage` → (`Pass` | `GPUPass`).

- **`Compute`** owns a list of `Stage`s and a shared `AnyBag` storage. `execute()` runs every stage in order each frame. Returning the sentinel string `kComputeStopExecution` from a stage halts the loop.
- **`Stage`** is a named, ordered sequence of passes stored as `std::variant<unique_ptr<Pass>, unique_ptr<GPUPass>>`. Returning `kStageStopExecution` stops that stage and bubbles up.
- **`Pass`** is a pure-CPU leaf. Implements `initialize()` and `execute()`.
- **`GPUPass`** is a sibling class (NOT a subclass of `Pass`) for WebGPU compute work. Implements `initialize()` and `submitWork()`, which records into the pass's own encoder and submits it through `GPU::submitCommands(encoder_label, record)` (label convention: `LOG_ID " encoder"`) **without waiting**. `OrtPass` is a `GPUPass` too: ORT's WebGPU EP submits to the same device queue and flushes by the end of `Session::Run`. Its sibling `OrtCpuPass` is a plain `Pass` that runs the model on ORT's CPU EP (`OrtProvider::LoadModel(path, use_webgpu_ep = false)`), with tensors in host memory.
- **`CustomPass`** wraps an arbitrary CPU lambda; used for buffer clearing and one-off algorithm steps.

**Perf recording** (`compute/performance.hpp`): `PerfRecorder` (`Compute::getPerf()`) is thread-safe; each thread's
records are rooted at its thread-local id (`PerfRecorder::setThreadId`: `main` in `main.cpp`, `loop_closure` on the
loop closure thread; unnamed threads get `thread <n>`), and the open-scope path is per thread. `Compute::finalize()`
writes `perf_<date>.yaml` to `WSLAM_PERF_OUT_DIR` *after* the flushables, so threads they join are in it;
`initizalize()` writes `init_perflog.yaml` and clears the records. `-DENABLE_PERF_RECORDING=OFF` (default ON)
defines `NPERF` on `wslam_compute` (PUBLIC), which compiles recording out: callers still hold a recorder and open
scopes, nothing is timed and no perf file is written.

GPU waits are **batched per stage**: `GpuStage::execute()` walks its passes, calls `submitWork()` on every consecutive `GPUPass`, then does one `GPU::waitForSubmittedWork` (`OnSubmittedWorkDone`) for the whole run, under a single perf scope and timeout label. A CPU `Pass` between two GPU passes waits for the run before it first. Ordering between passes comes from the single device queue (submit order), not from a shared encoder — so a pass's queue writes (`queue.WriteTexture`) must be issued before its own `submitCommands`, and GPU passes must never wait on the queue themselves.

The `Awaiter` (in `compute/include/compute/awaiter.hpp`) splits async work into two methods that must not be mixed in one chain when `executeAll` uses a non-zero timeout: `runChecked(callback, label)` wraps a synchronous API call in `PushErrorScope`/`PopErrorScope` (producing `WaitListEvent` futures) for validation-prone init code, while `addFuture(factory, label)` records a `wgpu::Future` returned by the factory (queue-serial for `OnSubmittedWorkDone`/`MapAsync`, no error scopes). Dawn rejects a `WaitAny` whose future set mixes those two sources (`EventManager.cpp` "Mixed source waits with timeouts are not currently supported"). For the common "submit a command buffer and wait for it" pattern, call `GPU::submitAndWait(commands, label, timeout)` instead of assembling it from an `Awaiter` by hand.

### GPU memory model (`compute/include/compute/gpu.hpp`)

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

### Shared model types (`wslam/src/models.hpp`)

Cross-pass payload types live in `wslam/src/models.hpp`: `Feature`, `FeatureSet`, `MatchResult`, `RansacResult`, `TriangulationResult`, plus the factor-graph types `PoseId`, `LandmarkId`, `LandmarkObservation`, `MapDelta`, `KeyframePose`, `LandmarkEstimate`, `MapStats`, `MapSnapshot`. New cross-pass types belong here unless they pull in heavy headers (GTSAM types live in `wslam/src/mapping_state.hpp` instead, behind a pImpl-style boundary).

### Pass constructor convention

CPU passes (`Pass` subclasses) must NOT take a `std::shared_ptr<compute::GPU>` in their constructor unless they actually call GPU APIs. The handful that do read back from the GPU (e.g. `LoadDataCPUPass` calls `gpu_->readTexture` / `readBuffer`, `VisualizeDataPass` uses the device for Pangolin texture upload) store `gpu_` as a private field — they do not inherit it. This was tidied across the codebase in the `refactor/bundle-gpu-passes-execution` work; preserve the convention when adding new passes.

GPU passes (`GPUPass` subclasses) inherit `gpu_` from the base class.

### SLAM pipeline (`wslam/`)

`CreateWslamPipeline()` in `wslam/include/wslam/wslam.hpp` assembles the pipeline from a `WslamConfig` (defined in `common.hpp`) that controls `enable_gui`, `max_iterations`, and `map_out_path`. It returns `std::optional<std::string>` (`std::nullopt` on success, an error message otherwise). `MappingState` is not handed back to the caller: it is owned jointly by the mapping passes via a `shared_ptr`, so it stays alive as long as the mapping stage lives inside `Compute`. The pipeline registers two callbacks with `Compute::addFlushable` that run in registration order inside `Compute::finalize()` (which `main.cpp` calls after the loop): first the mapping stage's `flush_async`, which blocks until any in-flight iSAM2 worker job completes so the final keyframe's optimisation lands in `MapSnapshotName`, then `ExportMap` (only when `map_out_path` is non-empty).

0. **Clear bindings stage** — a single `CustomPass` runs first every frame; calls `clearBuffersAndOffsets()` to zero non-retained GPU regions before new work is dispatched.

1. **Feature detect stage** (`CreateFeatureDetectStage`):
   - `SensorLoaderPass` (CPU) — pulls the next frame from the `std::generator` data provider, loads it (frames arrive as `data::PendingFrame`s and are only read from disk when kept), publishes the FrameBW to AnyBag, accumulates IMU samples into a vector under `GetImuVecName()`, and publishes the current frame's timestamp under `FrameTimestampNsName`. `kInitialFramesToSkip` is **0**: the stationary lead-in of the dataset is required — gravity and gyro bias are estimated from it and the first keyframe's zero-velocity prior assumes it (a previous 200-frame skip jumped past V1_01_easy's takeoff and forced a mid-flight gravity estimate).
   - `FillPyramidPass` (GPU) — builds a 6-level LoD image pyramid (scale 1.2× per level, hardcoded for 752×480).
   - `PassDetectCorners` + `CullCornersPass` (GPU) — FAST-style corner detection and NMS.
   - `GenerateFeaturesPass` (GPU) — computes ORB-style BRIEF descriptors (bit patterns from `brief_tests.inc`).
   - `LoadDataCPUPass` runs in the *next* stage (see below); the feature detect stage ends after `GenerateFeaturesPass` so all GPU work batches into a single submission.

2. **Pose estimate stage** (`CreatePoseEstimateCPUStage`):
   - `LoadDataCPUPass` — reads features (and, GUI only, LoD textures) back to CPU via `Awaiter`. Maintains the **reference feature set** (`FeatureSet(1)`): pinned to the last accepted keyframe's features and advanced only when the keyframe gate publishes the consume-once `FeatureReferenceAdvanceName` marker (every frame pre-origin, acceptance only after). All downstream matching/geometry therefore spans *previous keyframe → current frame* — exactly the factor-graph edge.
   - `MatchFeaturesCPU` — BRIEF descriptor matching (reference ↔ current) on CPU with mutual cross-check, Lowe ratio test at **0.85** (looser than textbook 0.75: V101's repetitive texture makes tight ratios reject true matches quasi-randomly, starving re-observation), and a 10%-frame spatial gate applied *before* the Hamming distance. The per-LOD candidate pool is the same LOD ± 1 (same-LOD inclusion is essential; an earlier version searched only neighbouring LODs and capped the match yield at the few % of scale-flickering features). Missing reference (first frame) stops only the stage (`kStageStopExecution`) so the mapping stage still bootstraps.
   - `RansacCPU` — normalised 8-point fundamental-matrix RANSAC with a Sampson-distance inlier test.
   - `TriangulateCPU` — undistorts the RANSAC inliers (scaled to LOD-0 pixels with `map::util::ToLod0Pixel`, which accounts for `Feature::lod`; it used to ignore it, which roughly doubled v1's rotation error), re-fits the essential matrix on calibrated rays, decomposes into four (R, t) candidates, picks the one with the most cheirality passes, then runs linear DLT triangulation and a reprojection-error filter.
   - `VisualizeDataPass` (when `enable_gui`) — Pangolin GUI for matches, RANSAC inliers, and projected landmarks.

3. **Mapping stage** (`CreateMappingStage` in `wslam/src/map/mapping.hpp`) — incremental visual-inertial SLAM via GTSAM iSAM2 with **smart projection factors**. State (the iSAM2 instance, the cached `Cal3_S2`, pose / landmark counters, the `flat_map<Feature, LandmarkId>` active-track map, IMU bias / velocity propagation, gravity estimate) lives in `MappingState`, injected by reference into all four passes. The frame-vs-keyframe geometry conflation that originally capped accuracy is fixed by the reference-feature-set mechanism above. Stage order:
   - `Isam2DrainPass` — runs **first** so the keyframe gate and factor builder see up-to-date pose / smart-factor estimates from the previous iSAM round. Without this, the factor builder would read stale `smart_factor_indices` and schedule removal of factor indices iSAM had already replaced.
   - `KeyframeGatePass` — gates on min-landmarks, min-rotation OR min-parallax, and a min count of *new* landmarks. Allocates a new `PoseId`; chains the world pose off the previous keyframe's `predicted_values` (NOT `latest_values` — the previous iSAM update may still be in-flight on the worker). Associates **all RANSAC inlier pairs** (not just triangulated landmarks — smart factors need no pre-validated 3D point, and associating on the triangulated subset capped re-observation at chance level) against the previous keyframe's `feat_curr → LandmarkId` map. Bootstraps the first keyframe at **motion onset** (small `bootstrap_parallax_px` vs the per-frame-advancing reference), then initialises gravity *and gyro bias* from the quietest stationary window found in the IMU buffer (min `std(|accel|) + mean gyro deviation` score — absolute thresholds fail: EuRoC's gyro bias is ~0.08 rad/s and a parked MAV vibrates).
   - `FactorBuilderPass` — constructs the per-frame `NonlinearFactorGraph` and `Values` delta: `PriorFactor<Pose3>` (gauge) on the first keyframe, `BetweenFactor<Pose3>` (vision-only odometry, with the inverse-transpose direction conversion that GTSAM's between convention requires), `CombinedImuFactor` between consecutive keyframes, priors on initial velocity and IMU bias, and one `SmartProjectionPoseFactor<Cal3_S2>` per landmark. Smart factors marginalise the 3D point out of the optimisation, sidestepping iSAM2's depth-axis singularities on low-parallax landmarks. Observations are **pre-undistorted upstream** (in the gate pass) so the smart factor sees a clean pinhole `Cal3_S2`. On re-observations, the existing smart factor is mutated (`add()`) and re-pushed to iSAM2 via the remove-and-readd pattern, with the old factor index drained out of `smart_factor_indices`.
   - `Isam2UpdatePass` — submits `(new_factors, new_values, remove_indices)` to an `Isam2Worker` running on a dedicated thread. The pass is asynchronous: each `execute()` drains the *previous* frame's pending result (blocking only if iSAM hasn't caught up), applies it to `MappingState::latest_values`, publishes a GTSAM-free `MapSnapshot` under `MapSnapshotName`, then submits the current frame's work. Catches `IndeterminantLinearSystemException` with the offending key formatted in the error message. Drained explicitly by `Isam2DrainPass` at the top of the next frame's stage; the mapping stage's `flush_async` (registered as a `Compute` flushable) runs once inside `finalize()` after the main loop exits to absorb the final keyframe's optimisation.

4. **Map export** (`wslam/src/export.hpp`, registered as a `Compute` flushable that runs inside `finalize()` *after* the pipeline loop exits, not as a per-frame pass): writes `MapSnapshot` and `CamSensorParams` to `<stem>.ply` (landmark point cloud, ASCII PLY with `x`, `y`, `z`, `landmark_id`) and `<stem>.json` (keyframes, intrinsics, stats, schema). Triggered only when `config.map_out_path` is non-empty.

### Python presenter (`presentor/`)

`presentor/` is a small PEP-621 `pyproject.toml` package providing the `presenter` CLI (`python -m presenter` or the `presenter` entry point). Dependencies: `rerun-sdk` (visualisation), `plyfile` (PLY loader), `numpy`. The viewer logs the landmark cloud, keyframe frustums, and the trajectory polyline into rerun.

`viewer.py` is split into `init` / `log_landmarks` / `log_camera` / `log_trajectory` helpers so the same pieces can drive an incremental loader later (`rr.set_time_sequence(...)` per arriving keyframe). The schema the loader expects is documented in `wslam/src/export.cpp` (`format_version = 1`).

### Benchmarks (`benchmarks/`)

Accuracy evaluation against ORB-SLAM3 and EuRoC ground truth using the
[evo](https://github.com/MichaelGrupp/evo) toolset. `benchmarks/ORB_SLAM3/` is
a **git subtree** (squashed) of upstream `UZ-SLAMLab/ORB_SLAM3` `master` with
two local patches (C++17 bump; `LoopClosing.h` `mnFullBAIdx` bool→int) — when
modifying it, remember the changes are tracked in this repo's history.

- `benchmarks/build_orbslam3.sh` — builds ORB-SLAM3 + its Thirdparty against
  OpenCV 4 and the project's `vendor/pangolin/install` (needs
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` under CMake ≥ 4). The system OpenCV is 5 now, which
  ORB-SLAM3 does not build against: `ORB_OPENCV_DIR` points it at an OpenCV 4.13 install
  (a minimal one lives in `~/dev/proj/resources/res_uni_slam/opencv4/install`), baked into
  the run path; `ORB_BUILD_JOBS` (8) caps the jobs.
- `benchmarks/run_orbslam3_euroc.sh [MAX_FRAMES] [RUN_INDEX]` — runs mono-inertial on the
  `EUROC_DIR` (or `.env`) sequence, read from the directory name (`V1_03_difficult` →
  `V103` timestamps); trajectories land in `benchmarks/results/` as
  `f_dataset-<SEQ>_monoi[_<N>f][_r<I>].txt`. ORB-SLAM3 is not deterministic: repeat it
  with a run index. The binary segfaults at shutdown *after* saving (known upstream
  viewer/GL teardown race) — the script tolerates it and validates the output files instead.
- `benchmarks/run_wslam_euroc.sh MAV0_DIR OUT_DIR [BUILD_DIR]` — a full v2 run
  (`--device=cpu`, mapping and loop closure on) over a sequence, `--max-iters` set to its
  cam0 frame count; `OUT_DIR` gets the maps, `run.log`, the perf yaml and `status.txt`.
  Don't edit it while a run uses it: bash reads scripts as it goes.
- `benchmarks/compare.py` (run with `benchmarks/.venv/bin/python`; venv has
  `evo` + `numpy`) — converts wslam `map.json`, ORB-SLAM3 txt (ns
  timestamps), and EuRoC GT csv to TUM format and runs `evo_ape`/`evo_rpe`.
  `--cam-to-body mav0/cam0/sensor.yaml` converts wslam's camera poses to the body
  frame (needed for a meaningful RPE); `--wslam` is repeatable (`map.json` and
  `map_vio.json` of a loop closure run). `--report` adds `accuracy_report.py`
  (numpy/scipy, also standalone): coverage, Umeyama scale, APE under Sim(3) / SE(3) /
  4-DoF position+yaw (needs the map json's `gravity_world`), rotation APE, RPE over
  1/5/10 m (metric and scaled), local scale spread over 5 s windows, gravity error at
  the first keyframe, APE-minimising time offset and loop-pair residuals; written to
  `<out-dir>/accuracy_report.md`. EuRoC's ground-truth body frame is ~5.5° off both
  ORB-SLAM3's and wslam's: the report estimates and removes that constant offset before
  rotation APE and RPE. evo's RPE does not, so its RPE reads ~4× too high (ORB-SLAM3:
  8.2 % vs 2.1 % over 1 m).
- `benchmarks/NOTE_HARD_SEQUENCES.md` — the current end-to-end numbers: wslam (VIO, pose
  graph, global BA) against ORB-SLAM3 (3 runs) on V1_01_easy, V1_03_difficult and
  V2_03_difficult, before and after the IMU-init change. `benchmarks/results/`
  (gitignored) holds the runs (`<SEQ>/wslam_{before,after}/`, ORB `f_`/`kf_` files) and
  their `eval/accuracy_report.md`; `summarize_reports.py` tabulates them. Datasets:
  `~/dev/proj/resources/res_uni_slam/euroc/vicon_room{1,2}/<sequence>/<sequence>/mav0`
  (V2_03 came from the Research Collection's `vicon_room2.zip` by HTTP range requests —
  the server throttles each connection to ~100 KB/s).
- `benchmarks/HOWTO_REEVALUATE.md` — the re-evaluation checklist: run with a
  per-experiment `--out-dir`, check Umeyama scale → RPE → APE in that order,
  and verify root-cause fixes with the debug-log greps (re-observation rate,
  `metric_step`, gravity line).

### v2 pipeline: ONNX feature detect (`wslam/src/feature_detect_v2/`)

`CreateWslamPipelineV2()` (selected with `-v2`) builds a standalone pipeline of three
stages: feature detect, pose estimate and mapping (below; `WslamConfig::enable_mapping`,
`-no-mapping` drops it). The `GpuStage` `v2::CreateFeatureDetectStage` runs the **split**
RaCo-ALIKED-LightGlue models (`resources/*.onnx`, gitignored; LightGlue-ONNX v3.0, 512
keypoints): an **extractor** (image → keypoints + 128-d descriptors) run once on the first
frame, then released (`SplitModelRunner`, `split_model.*`) and replaced by the **cached**
model (current image + the reference frame's cached keypoints/descriptors → the current
keypoints/descriptors and zero-padded `[K, 3]` matches/`[K]` scores, `mscores > 0` marks
real rows, `ParseMatches`). Features are never re-detected: each run's output is kept as
*pending* and becomes the reference when its frame does, so the reference keypoints are
exactly those the keyframe reported, under the same indices. All v2 code lives in
`namespace wslam::v2` (the v2 `FillPyramidPass`/`SensorLoaderPass` share names with v1 and
are compiled into the same library). Per frame: `SensorLoaderPass` → `FramePrefilterPass`
(below; with mapping) → `FillPyramidPass` (the frame's pyramid) → `PackModelInputPass`
(LoD0, centre-cropped to 736×480, gray → RGB, NCHW `[1, 3, H, W]` into the images buffer) →
`OrtCpuFeatureDetectPass` (`--device=cpu`: reads the images buffer back, CPU EP) or
`OrtFeatureDetectPass` (GPU pass in the same wait batch, WebGPU EP reading the images buffer
in place; blocked at startup until WebGPU exports exist). Both delegate to
`SplitModelRunner`, which publishes the matches, `OrtKeypointsName` (reference keypoints then
current, the `[left K | right K]` layout `LoadMatchesPass` reads) and the current frame's
`FrameFeatures` under `OrtFrameFeaturesName` (`shared_ptr<const>`, what loop closure stores).
Model validation and match parsing live in `onnx_utils.*`. Which frame is the reference is
chosen by `v2::ReferenceAdvance`: `EveryFrame` (no mapping: consecutive frames) or
`OnKeyframeMarker` (with mapping: it advances only when the mapping stage set
`FeatureReferenceAdvanceName` — every frame before the first keyframe, then on accepted
keyframes — so the model always pairs the last keyframe with the current frame, exactly the
factor-graph edge). The runner consumes the marker at the start of the next model run,
because acceptance is decided after the detect stage. The tensor names, shapes and row
layouts are in `raco_lightglue.hpp`; outputs land in AnyBag under the
`ResourceIdentifier::Ort*` keys. The keypoint count is a runtime value
(`WSLAM_ONNX_KEYPOINTS`) carried by `v2::SharedBindings::getKeypointCount()`, and the
keypoint-dependent parts of the contract are functions of it (`KeypointsShape`,
`DescriptorsShape`, ...).

**Frame prefilter** (`frame_prefilter.*`, `motion_prediction.*`): with mapping, a CPU pass
before any GPU work skips the model (`kStageStopExecution` + the `FrameSkippedName`
marker) on frames whose motion since the last keyframe the IMU predicts, with confidence,
to fail the gate's small-motion test (rotation < `min_rotation_rad` AND median parallax <
`min_parallax_px`). The mapping stage's `BuildFactorsPass` publishes a
`KeyframeMotionPrior` per accepted keyframe with an IMU edge (IMU-predicted orientation and
velocity, biases, gravity, median landmark depth = triangulated unit depth × the IMU
`metric_step`, the gate thresholds, and a velocity sigma measured online: the drain
compares each keyframe's predicted velocity with the smoother's, an EMA in
`SharedBindings::velocity_error_variance`); the gate erases it on acceptance, so there is
no skipping before the second keyframe. `MotionPredictor::predict` preintegrates the buffered
samples after the keyframe with the mapping stage's own calibration, predicts the median
parallax over an image grid at the median depth, and bounds both quantities by
`confidence_sigmas` of the gyro/accel noise and random walk (`IMUSensorParams`), bias,
attitude, velocity and depth errors (default 2σ). The parallax bound is the largest median
parallax over the motions with each error at kσ along 14 sampled directions (axes and
diagonals) and the near depth — not the median plus a corner-pixel worst case, which was
2–3× looser. `max_consecutive_skips` (10) forces a model run. Before there is a prior
(no keyframe yet) it skips frames whose last 50 IMU samples are stationary
(`map::utils::StationaryScore` < 0.5; the first frame always runs, at most 40 in a row:
the score can't see constant rotation or velocity) — V1_01_easy's parked lead-in is only
~100 frames, the MAV lifts off around frame 105.
Skipped frames leave the marker and the pending features for the next run, so promotion
stays correct. The gate logs predicted vs measured motion on frames that ran
(`grep '\[prefilter\]'`).

`v2::CreatePoseEstimateStage` (`wslam/src/pose_estimate_v2/`, CPU `Stage`) is the
port of v1's pose estimate stage; the model already matched, so there is no matcher:
- `LoadMatchesPass` — takes the published keypoints and joins them with the model's
  matches (taken, consume-once) into `v2::KeypointCorrespondence`s: LOD-0 source pixels
  (`raco_lightglue::ToSourcePixel` adds the crop offset and removes the model's
  pixel-centre 0.5) plus undistorted rays. Stops the stage on frames the model did not
  run on (the first, and prefilter skips).
- `RansacPass` — drops matches below a LightGlue score floor (`MatchFilterOptions`,
  0.5 with a top-100 fallback; the model itself already cuts at 0.1), then runs
  eight-point **essential**-matrix RANSAC on the undistorted rays (the camera is
  calibrated, so there is no F) with a Sampson test scaled to pixels by the focal
  length. The iteration count follows `log(1 - p) / log(1 - w^8)` for the best inlier
  ratio `w` (`util::RansacOptions`; `max_iterations` is only a guard); on V101
  keyframes it draws ~5 samples. The final least-squares re-fit on the inliers is
  weighted by the match scores (`ScoreWeighting`).
- `TriangulatePass` — decomposes RANSAC's E (cheirality pick), refines (R, t) by
  score-weighted Huber Levenberg-Marquardt on the Sampson error
  (`util::RefineRelativePose`; this, not the floor or the linear weights, is where most
  of the accuracy comes from), then DLT and filters.
The pass bodies are also free functions (`v2::RunRansac`, `v2::RunTriangulation`) so
`PoseEstimateDatasetTest.V2MatchScoreAblation` can re-score configs on one set of
recorded model matches. The geometry is pure functions in `pose_estimate_v2/util/`
(`epipolar_geometry`, `camera`). Payload types (`KeypointMatches`, `KeypointRansacResult`,
`KeypointTriangulationResult`) are in `models.hpp`, keys `ResourceIdentifier::Keypoint*`.
The model's keypoints come out of a sorted `TopK`, so a keypoint's index is its RaCo
rank; the v2 mapping stage relies on it being deterministic (below).
The GUI is not ported to v2.

`v2::mapping::CreateMappingStage` (`wslam/src/map_v2/`, `namespace wslam::v2::mapping`
— not `v2::map`, which would shadow `wslam::map` inside v2 code) ports the v1 mapping
rewrite to the model's correspondences and replaces iSAM2 with a sliding-window
`gtsam::BatchFixedLagSmoother`. Graph and priors are v1's (smart projection factors,
`CombinedImuFactor`, no visual between factor, ids from 1, hybrid initial guess);
shared IMU helpers and env knobs live in `map/utils.{hpp,cpp}`. Stage order
**Filter → Drain → Associate → Build → Submit**:
- `FilterKeyframePass` — v1's gate (min landmarks, rotation OR median parallax over all
  inliers, min new landmarks, bootstrap/gravity window). *Takes* the triangulation and
  RANSAC results so an early-stopped pose estimate never maps stale geometry. Publishes
  the reference marker and a `KeyframeCandidate`. The first keyframe (the origin) needs
  only the bootstrap tests (gravity window, `bootstrap_min_parallax_px` frame to frame,
  min landmarks): applying the small-motion test to frame-to-frame motion held it back
  to frame ~670 of V1_01_easy, 27 s into the flight, where the zero-velocity prior was
  wrong and the stationary window's gravity was 8° off; it now lands at liftoff
  (frame ~107). IMU initialisation (`initializeFirstFrame`): gravity and gyro bias come
  from the quietest window (`map::utils::RecoverStartupImuState`: gravity of nominal
  magnitude along the mean specific force, the excess along it is accel bias — the
  part across gravity can't be told from tilt at rest). The **last** window still below
  `zupt_max_score` marks the last rest (motors spinning up still count): the first
  keyframe's velocity is preintegrated from there (`map::utils::PropagateFromRest`, which
  also turns gravity into the keyframe's camera) instead of being assumed zero, with a
  prior sigma that grows with the integrated time. `BuildFactorsPass` puts x1's pose prior
  tight on position and on rotation about gravity only: its tilt gets
  `prior_pose_tilt_sigma_rad` (3°), so the smoother re-levels gravity jointly with the
  accel bias once the body rotates (VIO's gauge is 4-DoF). Score the result with the
  report's **world-frame** gravity line: the camera-frame one at x1 includes EuRoC's
  ~5.5° ground-truth body-frame offset.
- `DrainSmootherPass` → `SubmitSmootherPass::drainPending()` — blocks on the previous
  keyframe's update (only accepted keyframes reach it), archives keyframes that left
  the window (`SharedBindings::archived_poses`, last in-window estimate), and *retires*
  every landmark whose smart factor the smoother marginalised (its info is in the
  marginal; last triangulation goes to `archived_landmarks`). The archive is append-only
  and is the input meant for later re-observation handling on another thread.
- `AssociateLandmarksPass` — after the drain, so it sees retirements. Tracks are keyed
  by the model's keypoint index in the last keyframe image (`active_landmarks`) and
  confirmed by pixel (`LookupTrack`, 0.5 px; with cached features the reference keypoints
  are the keyframe's own, so a mismatch is a bug). A match continues a landmark only if its factor is
  still in the window; otherwise it starts a new one (prev + curr observations).
- `BuildFactorsPass` — factors + values, every new key stamped with its keyframe index
  (`SmootherTime`); re-observed smart factors are cloned, old slot in `remove_slots`.
- `SubmitSmootherPass` — submits to `SmootherWorker` without waiting.
`SmootherWorker` (`smoother_worker.hpp`) owns a `LocalSmoother` (a
`BatchFixedLagSmoother` whose `update(..., remove_slots)` frees slots through the
protected `removeFactors()`: the base `factorsToRemove` only nulls them, which leaks one
slot per re-observation) on a `base::JobWorker` thread (one job in flight). The window is
counted in keyframes (`WSLAM_SMOOTHER_WINDOW`, default 10; lag = N − 1);
`WSLAM_SMOOTHER_MAX_ITERS` (default 10) caps LM per update. The base API returns no
factor indices, so the result carries `live_slots` (factor pointer → slot); a submitted
smart factor missing from it was marginalised. Never mutate a submitted factor or call
`point()` on one while a job runs. Snapshot = archive + in-window estimate.

**Loop closure** (`wslam/src/map_v2/loop_closure/`, `namespace
wslam::v2::mapping::loop_closure`; results and process in
`benchmarks/NOTE_LOOP_CLOSURE.md`) runs on its own thread (`LoopClosureWorker`, a
mutex + condvar event queue — not `base::JobWorker`, which allows one job).
`FeedLoopClosurePass` runs last in the mapping stage (accepted keyframes only) and never
waits: it pushes the keyframe's image (`PlaceImage`, 8-bit) with the landmarks it tracks
(`active_landmarks` pixels), and a `MapUpdateEvent` with the pose the last drain settled
plus what the drains archived (`SharedBindings::*_since_feed`); it pulls the newest
`DriftCorrection` into `SharedBindings::drift_correction`. The worker embeds **every**
keyframe, gates it into the pose graph (`NodeGate`: a keyframe that did not move/turn
enough from the last node is attached to it, not added), adds nodes to a FAISS
`IndexIDMap2(IndexFlatIP)` and queries only the **newest** node of each batch:
top-N candidates at least `min_keyframe_gap` keyframes / `min_time_gap_s` older, then
the cached RaCo-LightGlue model on the query image against the candidate's stored features
(`CachedMatcher`, its own CPU session: each `KeyframeImageEvent` carries the front end's
`FrameFeatures` and its tracks by keypoint index), candidate keypoint index → landmark
(exact) → archived 3D points in the
candidate camera frame, then PnP (`pnp_verify.*`: own 6-point DLT RANSAC + Huber
Gauss-Newton, no OpenCV) and a consistency bound against the graph's estimate.
The pose graph (`pose_graph.*`) is pose-only iSAM2: odometry `BetweenFactor`s from the
VIO, a `Pose3AttitudeFactor` per node holding roll/pitch to the VIO (gravity observes
them), loop `BetweenFactor`s (Gaussian by default: a Cauchy kernel would ignore the
many-sigma correction a true loop makes). The **smoother is never modified** (its
marginals are linearised in the VIO frame); the correction is a 4-DoF drift transform
for live output, and at export `ApplyCorrections` (`merge.*`) moves every keyframe by its
node's `T_corr * T_vio^-1` and every landmark by its anchor keyframe's
(`SharedBindings::landmark_anchor`). The flush order is smoother flush → final feed →
`finish()` → merge → global BA: `MapSnapshotName` becomes the corrected map and
`MapSnapshotVioName` keeps the smoother's, exported as `<stem>_vio.{ply,json}` (so one
run yields both trajectories). **Global BA** (`global_ba.*`, `MergeLoopClosure` in
`map_stage.cpp`): the whole run re-solved from the corrected map with GTSAM LM —
explicit `Point3` landmarks with Huber `GenericProjectionFactor`s (not smart factors:
their zero-on-degeneracy is what collapsed scale in an earlier global batch BA),
`CombinedImuFactor`s re-preintegrated from the samples `SharedBindings::imu_edges` kept,
velocity/bias from `archived_navigation`, every observation (`landmark_observations`),
and each accepted loop's PnP inliers as observations of the candidate's landmarks from
the query keyframe (`AcceptedLoop::observations`) — what lets it fix more than the
pose graph's 4 DoF. Landmarks need ≥ 3 views, positive depth and rays ≥ 1° apart (lower
parallax made GTSAM report indeterminate Hessians). Two solves (outliers > 4 px dropped
in between, the second capped at a third of the iterations), a tight prior on the first
keyframe; rejected (pose graph map kept) when the error does not drop or the path length
changes by > 5 %. Single-threaded (GTSAM is built without TBB): 200–340 s on the 2-minute
EuRoC sequences (~700–1200 keyframes, 300–370k observations), so it fits only at export —
ORB-SLAM3 runs its full BA in a thread but only below 200 keyframes
(`LoopClosing.cc:1199`), otherwise it relies on its essential graph. Results:
`benchmarks/NOTE_HARD_SEQUENCES.md`. When accepted, `MapSnapshotName` is the adjusted map and
`MapSnapshotPgoName` the pose graph's, exported as `<stem>_pgo.*`; the json's
`loop_closure.global_ba` reports it. Models: `place_recognition.*` (MegaLoc / MixVPR, contract
checked at load, `model_io.*` adapts images by **squashing** to the model size with exact
area averaging, gray → RGB, ImageNet normalisation — neither export normalises). With
`--device=gpu` the place recognition session runs on the WebGPU EP on an auxiliary
device (`GPU::createAuxiliaryDevice`, EP context 2: the pipeline's device is not shared
across threads); the matcher stays on the CPU (`matcher_on_gpu`). Loop closure sessions
use `WSLAM_LC_THREADS` (4) threads with spinning off.

`frontend_recording.{hpp,cpp}` (`v2::FrontendRecorder` / `v2::FrontendReplay`,
MessagePack): with mapping the front end depends on the keyframe decisions, so a
recording is a golden run — per frame the model's matches, new IMU samples and whether
mapping advanced the reference, plus the run's final keyframes.

ORT interop (`compute/onnx_interop.hpp`) facts that are easy to get wrong:
- An external device needs `deviceId` > 0 (`OrtProvider::kWebGpuContextId` = 1)
  plus the **raw** `WGPUInstance`/`WGPUDevice` handle values.
- The EP's `OrtDevice` id is always 0 whatever the context id, so tensors in
  device memory use `Ort::MemoryInfo("WebGPU_Buf", ..., 0, ...)`.
- ORT binds a tensor's buffer whole from offset 0, so model tensors can't live
  in `GPU` slab sub-regions: they use `GPU::createDedicatedBuffer` (GPU-owned).
- The split models' outputs all have fixed shapes (matches are zero-padded to K
  rows), so they are bound to host tensors; a data-dependent output would have to
  be rebound by memory info before every run, or ORT reuses the previous run's
  output as a pre-allocated value and rejects the new shape.
- When ORT is handed our device it only gets the limits/features `GPU::initDevice`
  requested, which therefore mirrors what ORT would request for its own device.

### Data providers (`data/`)

Providers implement `ProviderBase<N>` and expose a `std::generator<std::expected<Reading<N>, std::string>> getReadings()` coroutine. A reading's frames are `PendingFrame`s (timestamp + `load()`), so nothing is read from disk for frames a consumer drops (e.g. cam1 under `AdaptProvider<1>`). `AdaptProvider<Requested, Available>` adapts an N-camera provider to an M-camera consumer. Currently only `EurocProvider` is provided; the earlier `TumProvider` has been removed.

Both `SensorLoaderPass`es take an optional `data::ReadingFilter`: it sees every frame and IMU sample (`ReadingCandidate`: kind, timestamp, per-kind index) *before* any file is read and returns `Keep`, `Skip` or `StopStream`. `MakeKeyframeFilter` (`wslam/src/keyframe_schedule.hpp`) keeps all IMU and only a schedule's frames; a lead-in skip would skip both kinds. The loaders return `kComputeStopExecution` at the end of the stream; `Compute::execute` reports that stop as success, so callers can't tell the stream ended (hence `--max-iters`).

### Error handling convention

- Functions that can fail return `std::optional<std::string>` — `std::nullopt` means success, a string is the error message.
- Functions that return values use `std::expected<T, std::string>`.
- WGSL shaders are loaded and validated at `initialize()` time, not at dispatch time.
- Per-pass logs use a `LOG_ID` macro (`#define LOG_ID "[Pass Name]"`) at the top of each `.cpp`; all spdlog calls prefix with it for grep-friendly timelines.
