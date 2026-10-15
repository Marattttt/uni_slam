# NOTE: hard EuRoC sequences — wslam vs ORB-SLAM3, IMU init and global BA

Measured 2026-10-15 on `feat/v2-frame-prefilter`. "Before" is tag
`bench/imu-baseline` (496946d); "after" adds the IMU-initialisation change and the
export-time global bundle adjustment (see `CLAUDE.md`, v2 mapping: `FilterKeyframePass`
and **Global BA**).

## TL;DR

- **Robustness:** both systems complete V1_03_difficult and V2_03_difficult on every
  run. wslam initialises once and never resets. ORB-SLAM3 mono-inertial resets its map
  1–3 times during initialisation (up to 47 frames lost) before holding one map; it
  ended with a single map in all 6 runs.
- **Accuracy:** ORB-SLAM3 is still about 2× better on the hard sequences.
  - V1_03: 0.017–0.026 m APE vs wslam 0.041 m.
  - V2_03: 0.018–0.020 m vs 0.046 m.
  - V1_01_easy: close, 0.036 m vs 0.044 m, with equal RPE over 5 and 10 m.
- **The global BA is the large improvement.** It cuts wslam's APE 3–4× against the VIO
  (and against the pose graph) on all three sequences, and roughly halves local error:
  RPE over 1 m and the 5 s local-scale spread.
  - The old pose-graph-only correction made local accuracy *worse* than the raw VIO:
    RPE over 1 m rose from 4–5 % to 5–9 %.
  - The BA brings it down to 2.4–2.7 %.
- **The IMU-initialisation change is neutral on accuracy** (VIO APE within ±6 %, no
  consistent sign).
  - Its velocity estimate at x1 is right: 0.077 m/s against ground truth 0.080 m/s on
    V1_03, where the old prior assumed 0.
  - The baseline's gravity was already good once measured properly: 0.45° world-frame
    error on V1_01. The "2.9° at the first keyframe" figure was EuRoC's ~5.5°
    ground-truth body-frame offset leaking into a camera-frame metric (fixed in the
    report).

## Setup

| | |
|---|---|
| Machine | 16 cores / 32 threads, no discrete GPU; `--device=cpu` |
| wslam | `benchmarks/run_wslam_euroc.sh` — v2, mapping + loop closure (MixVPR), Release build, whole sequence |
| ORB-SLAM3 | `benchmarks/run_orbslam3_euroc.sh 0 <run>` — mono-inertial, the vendored subtree rebuilt on OpenCV 4.13, 3 runs per hard sequence (V1_01: the existing single baseline run) |
| Scoring | `compare.py --report` → `accuracy_report.md`; `summarize_reports.py` for the tables. Sim(3)/SE(3) Umeyama alignment; rotation APE and RPE with the constant GT body-frame offset removed |
| wslam outputs | `map_vio` = smoother (VIO), `map_pgo` = pose-graph correction, `map` = global BA (after) or pose graph (before) |

Caveats:
- **Which poses are scored.** wslam is scored on keyframes, ORB-SLAM3 on every frame.
- **Coverage.** Both start a few seconds after the ground truth, and coverage is equal
  within 1 % on the hard sequences. On V1_01 ORB-SLAM3 starts 11.6 s in and wslam
  4.3 s in.
- **Timing.** wslam's loop closure depends on timing (queries batch on the loop
  thread), so its pose-graph numbers vary between runs. The VIO numbers do not.
- **Concurrent load.** Some baseline runs overlapped a compile.

## Robustness and cost

| | keyframes | loops | resets / frames lost | wall time |
|---|---|---|---|---|
| V1_03 wslam before / after | 1192 / 1193 | 149 / 145 accepted | 0 / 0 | 636 s / 971 s (BA 343 s) |
| V1_03 ORB r1 / r2 / r3 | 386 / 402 / 382 | 1 / 7 / 8 detected | 3/47, 3/27, 2/17 | real time |
| V2_03 wslam before / after | 1138 / 1138 | 121 / 107 accepted | 0 / 0 | 644 s / 885 s (BA 242 s) |
| V2_03 ORB r1 / r2 / r3 | 530 / 532 / 523 | 2 / 1 / 6 detected | 1/3 each | real time |
| V1_01 wslam before / after | 718 / 717 | 177 / 172 accepted | 0 / 0 | 598 s / 751 s (BA 200 s) |

The global BA ran on:

| | landmarks | observations | from loop PnP inliers |
|---|---|---|---|
| V1_01 | 38.8k | 293k | 25.8k |
| V1_03 | 57.4k | 370k | 10.9k |
| V2_03 | 47.0k | 283k | 8.8k |

ORB-SLAM3's maps stay above 200 keyframes, so its loop closures never ran a full BA
(`LoopClosing.cc:1199`); it relied on the essential graph.

## Accuracy

### V1_03_difficult

| metric | ORB r1 | ORB r2 | ORB r3 | VIO before | PGO before | VIO after | PGO after | **GBA after** |
|---|---|---|---|---|---|---|---|---|
| Umeyama scale | +1.69% | +0.06% | +1.82% | +5.91% | +2.41% | +5.12% | +0.95% | +2.25% |
| APE Sim(3) RMSE | 0.026 m | 0.017 m | 0.026 m | 0.156 m | 0.166 m | 0.161 m | 0.116 m | **0.041 m** |
| APE SE(3) RMSE | 0.037 m | 0.017 m | 0.038 m | 0.175 m | 0.169 m | 0.175 m | 0.117 m | 0.051 m |
| APE Sim(3) max | 0.282 | 0.220 | 0.190 | 0.299 | 0.539 | 0.340 | 0.415 | 0.097 |
| gravity error (world) | – | – | – | 1.45° | 2.44° | 1.64° | 1.42° | 0.81° |
| rotation APE | 0.58° | 0.51° | 0.61° | 3.00° | 3.29° | 3.01° | 1.86° | 1.01° |
| RPE 1 m (median) | 1.4% | 1.0% | 1.5% | 4.2% | 7.5% | 4.4% | 5.9% | 2.6% |
| RPE 5 m (median) | 0.5% | 0.4% | 0.5% | 2.4% | 3.0% | 2.4% | 2.4% | 1.1% |
| RPE 10 m (median) | 0.3% | 0.2% | 0.3% | 2.2% | 2.0% | 2.1% | 1.4% | 0.7% |
| local scale spread (5 s) | 3.1% | 1.5% | 3.0% | 10.1% | 16.2% | 10.7% | 14.2% | 5.0% |

### V2_03_difficult

| metric | ORB r1 | ORB r2 | ORB r3 | VIO before | PGO before | VIO after | PGO after | **GBA after** |
|---|---|---|---|---|---|---|---|---|
| Umeyama scale | −0.14% | −0.32% | −0.36% | −0.94% | +6.85% | −1.13% | +4.83% | +0.57% |
| APE Sim(3) RMSE | 0.020 m | 0.020 m | 0.018 m | 0.176 m | 0.198 m | 0.168 m | 0.191 m | **0.046 m** |
| APE SE(3) RMSE | 0.020 m | 0.021 m | 0.019 m | 0.177 m | 0.235 m | 0.170 m | 0.212 m | 0.047 m |
| APE Sim(3) max | 0.159 | 0.164 | 0.214 | 0.294 | 0.562 | 0.273 | 0.509 | 0.144 |
| gravity error (world) | – | – | – | 2.24° | 1.58° | 2.42° | 0.19° | 0.61° |
| rotation APE | 0.45° | 0.48° | 0.44° | 2.34° | 3.33° | 2.75° | 3.13° | 0.58° |
| RPE 1 m (median) | 1.4% | 1.4% | 1.3% | 5.0% | 8.3% | 5.1% | 8.7% | 2.7% |
| RPE 5 m (median) | 0.5% | 0.5% | 0.5% | 2.3% | 5.6% | 2.3% | 5.3% | 1.1% |
| RPE 10 m (median) | 0.3% | 0.3% | 0.3% | 1.7% | 3.0% | 1.8% | 2.8% | 0.6% |
| local scale spread (5 s) | 2.5% | 2.3% | 2.6% | 18.2% | 25.4% | 12.4% | 32.6% | 7.2% |

### V1_01_easy (regression check)

| metric | ORB | VIO before | PGO before | VIO after | PGO after | **GBA after** |
|---|---|---|---|---|---|---|
| Umeyama scale | +1.33% | −0.67% | +2.15% | −0.85% | +1.81% | +0.29% |
| APE Sim(3) RMSE | 0.036 m | 0.124 m | 0.072 m | 0.132 m | 0.072 m | **0.044 m** |
| APE SE(3) RMSE | 0.043 m | 0.124 m | 0.081 m | 0.133 m | 0.079 m | 0.044 m |
| gravity error (world) | – | 0.45° | 0.56° | 0.34° | 0.38° | 0.60° |
| rotation APE | 0.59° | 0.86° | 1.09° | 0.88° | 1.08° | 0.82° |
| RPE 1 m (median) | 2.1% | 3.5% | 5.3% | 3.5% | 5.2% | 2.4% |
| RPE 5 m (median) | 1.3% | 1.7% | 2.3% | 1.7% | 2.4% | 1.2% |
| RPE 10 m (median) | 0.6% | 0.9% | 1.1% | 0.9% | 1.0% | 0.6% |
| local scale spread (5 s) | 3.2% | 12.1% | 13.6% | 9.9% | 12.4% | 4.8% |

## What changed and what it did

### IMU initialisation

What changed:
- **x1 velocity.** It is preintegrated from the last rest window (zero-velocity
  update) instead of a zero prior.
- **Accelerometer bias.** Its component along gravity is split from |g|.
- **x1 tilt.** x1's prior is tight on position and yaw only; tilt gets 3°.

Measured:
- **The velocity prior is now right.** At V1_03's x1 the rest ended 1.0 s earlier; we
  integrate 0.077 m/s against GT 0.080 m/s.
- **The VIO doesn't measurably change.**

  | | V1_01 | V1_03 | V2_03 |
  |---|---|---|---|
  | VIO APE before → after | 0.124 → 0.132 m | 0.156 → 0.161 m | 0.176 → 0.168 m |
  | local scale spread before → after | 12.1 → 9.9 % | 10.1 → 10.7 % | 18.2 → 12.4 % |

  On the golden V1_01 1200-frame replay (deterministic, no loop closure) APE went
  0.044 → 0.040 m.
- **Why so little:** the old init was not as wrong as the camera-frame metric
  suggested. World-frame gravity error was 0.45° (V1_01) to 2.2° (V2_03). The VIO's
  remaining error is local scale wander (10–18 % over 5 s windows), which the first
  keyframe's state does not control.

### Global BA (export time)

**How it works:**
- Explicit landmarks with Huber reprojection factors, re-preintegrated
  `CombinedImuFactor`s over every keyframe edge, plus each loop's PnP inliers as extra
  observations.
- It starts from the pose-graph-corrected map.
- **Accepted on all three sequences.** Path length changed by −0.5 % to +1.3 %, with no
  scale collapse.

**What it did:**
- 3–4× lower APE than the VIO, and below the pose graph everywhere.
- RPE and local scale spread halved.
- Rotation APE down to 0.6–1.0°, near ORB-SLAM3's 0.4–0.6°.

**Bugs found on the way:**
- **Abort after a finished solve.** The first V1_03 run aborted: a landmark the outlier
  pass left with fewer than three views was looked up in the second solve's result.
  Fixed; `GlobalBaTest.DropsLandmarksTheOutliersLeaveTooFewViews` covers it.
- **Indeterminate Hessians.** Rays under 1° apart made GTSAM print them; those
  landmarks are now dropped.

## Why ORB-SLAM3 is still ~2× better on the hard sequences

- **Front-end density.**
  - wslam keeps 2–3× more keyframes (no keyframe culling) and tracks with a 512-keypoint
    learned matcher on keyframe pairs.
  - ORB-SLAM3 tracks every frame against a local map, with local BA over covisible
    keyframes.
  - wslam's VIO local scale spread of 10–18 % against ORB's 2–3 % is the clearest
    symptom. The BA only partly repairs it (5–7 %).
- **Online vs offline.** wslam's BA runs once, single-threaded, at export (200–340 s);
  live output is still the pose-graph map.
- **Candidates:**
  - a windowed local BA with explicit landmarks in the smoother (the scale spread
    problem);
  - keyframe culling (cost of everything downstream);
  - running the global BA on the loop thread, ORB-style (abortable, under a keyframe
    budget).

## Reproduce

```bash
E=~/dev/proj/resources/res_uni_slam/euroc
./benchmarks/run_wslam_euroc.sh $E/vicon_room1/V1_03_difficult/V1_03_difficult/mav0 \
    benchmarks/results/V103/wslam_after build/release
for i in 1 2 3; do
    EUROC_DIR=$E/vicon_room1/V1_03_difficult/V1_03_difficult/mav0 \
        ./benchmarks/run_orbslam3_euroc.sh 0 $i
done
benchmarks/.venv/bin/python benchmarks/compare.py --report \
    --wslam benchmarks/results/V103/wslam_after/map_vio.json \
    --wslam benchmarks/results/V103/wslam_after/map_pgo.json \
    --wslam benchmarks/results/V103/wslam_after/map.json \
    --gt $E/vicon_room1/V1_03_difficult/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv \
    --cam-to-body $E/vicon_room1/V1_03_difficult/V1_03_difficult/mav0/cam0/sensor.yaml \
    --out-dir benchmarks/results/V103/wslam_after/eval
```

The "before" binary: `git worktree add ../uni_slam_baseline bench/imu-baseline`, symlink
the three `vendor/*/install` directories, Release build, same runner with that build
directory.
