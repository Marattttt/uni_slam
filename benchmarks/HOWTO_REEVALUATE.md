# How to re-evaluate wslam accuracy

Checklist for re-running the evaluation after a change to the pipeline
(mapping stage, front-end, anything). Designed so each experiment leaves a
named, comparable artifact. Background and how to interpret the numbers:
`benchmarks/ACCURACY_ANALYSIS.md`.

All commands run **from the repo root**. Environment variables are passed
inline on each command (no `source .env` needed — the values below are the
.env ones):

```bash
EUROC=/home/marat/dev/proj/resources/res_uni_slam/euroc/vicon_room1/V1_01_easy/V1_01_easy/mav0
SHADERS="$PWD/resources"
```

## Fast iteration: 20% of the sequence

A full V101 run costs a lot of wall-clock for an iteration loop. For
development, run the first **582 frames (~20%, ≈29 s: stationary lead-in +
takeoff + first room sweep)** and compare both systems on the same segment:

```bash
# wslam, first 582 frames (Release build assumed, see §1):
EUROC_DIR=$EUROC WSLAM_SHADER_SRC_DIR=$SHADERS \
    ./build/pc_wslam --max-iters=582 --map-out=map.ply

# ORB-SLAM3 on the same 582 frames (truncated-timestamps limiter built
# into the runner; output suffixed _582f so the full baseline survives):
EUROC_DIR=$EUROC ./benchmarks/run_orbslam3_euroc.sh 582

# compare both against GT on the segment:
EUROC_DIR=$EUROC benchmarks/.venv/bin/python benchmarks/compare.py \
    --wslam map.json \
    --orb benchmarks/results/f_dataset-V101_monoi_582f.txt \
    --gt "$EUROC/state_groundtruth_estimate0/data.csv" \
    --out-dir benchmarks/results/<experiment-name>
```

Only compare equal segments against each other (582-frame vs 582-frame,
full vs full) — APE/RPE grow with trajectory length.

---

## 0. One-time prerequisites (already done; only redo if broken)

```bash
./benchmarks/build_orbslam3.sh                    # ORB-SLAM3 baseline binary
python -m venv benchmarks/.venv \
  && benchmarks/.venv/bin/pip install evo numpy   # evo toolset
benchmarks/.venv/bin/evo_config set plot_backend Agg   # no system Tk
```

The ORB-SLAM3 **baseline trajectory only needs to be regenerated when the
sequence changes** (it does not depend on wslam code):

```bash
./benchmarks/run_orbslam3_euroc.sh          # full sequence
./benchmarks/run_orbslam3_euroc.sh 582      # first 582 frames only
# -> benchmarks/results/f_dataset-V101_monoi[_582f].txt (+ kf_ variant)
# NOTE: the binary segfaults AFTER saving (known upstream shutdown race);
# the script tolerates it. Success = non-empty trajectory files.
# The frame limit works by truncating the EuRoC timestamps file the
# example iterates over — ORB-SLAM3 itself stays unpatched.
```

If a baseline already exists in `benchmarks/results/`, skip this.

## 1. Produce a fresh wslam trajectory

```bash
# Release build + full run + map export (map.ply + map.json in repo root):
./present_and_run.sh            # Ctrl-C the presenter when it opens, or:
./run.sh --map-out=map.ply      # Debug build — slower, use for sanity only
```

For accuracy numbers always use the **Release** build (`present_and_run.sh`
configures Release; `run.sh` configures Debug with sanitizers).

## 2. Convert + compare against ground truth

Give every experiment its own output directory — that is what makes runs
comparable later:

```bash
source .env
EXPERIMENT=2026-06-04_baseline        # name it after the change you made
benchmarks/.venv/bin/python benchmarks/compare.py \
    --wslam map.json \
    --orb benchmarks/results/f_dataset-V101_monoi.txt \
    --gt "$EUROC_DIR/state_groundtruth_estimate0/data.csv" \
    --out-dir "benchmarks/results/$EXPERIMENT"
```

This writes `wslam.tum`, `gt.tum`, the converted ORB trajectory, and prints
APE + RPE for both systems. `--convert-only` skips the evo runs if you only
want the `.tum` files.

## 3. Record the three headline numbers

For each experiment, note (a table in your lab notes / the paper is enough):

| number | command that prints it | current value (2026-06-04) |
|---|---|---|
| APE RMSE, Sim(3) | step 2 output (`evo_ape ... -as`) | 1.62 m |
| **Umeyama scale** | `evo_ape tum gt.tum wslam.tum -as -v` → "Scale correction" | **0.18** |
| RPE RMSE per 1 m | step 2 output (`evo_rpe ... --delta 1 --delta_unit m`) | 2.76 m |

Expected progression as fixes land — check them **in this order**:

1. **Umeyama scale → ~1.0** first. While it is far from 1, the trajectory
   shape is still uncorrelated with GT and the APE value is not very
   meaningful (the fit is degenerate). Don't celebrate APE changes yet.
2. **RPE drops** next — local odometry becoming correct.
3. **APE drops** last — global consistency (will plateau without loop
   closure; that plateau is the honest "current stage" number).

Also sanity-check `wslam.tum` itself: timestamps ~1.4e9 (seconds), quaternion
norms ~1, pose count ≈ keyframe count from the run log.

## 4. Pipeline-internal diagnostics (validates *which* root cause improved)

```bash
EUROC_DIR=$EUROC WSLAM_SHADER_SRC_DIR=$SHADERS SPDLOG_LEVEL=debug \
    ./build/pc_wslam --max-iters=582 > /tmp/wslam_debug_run.log 2>&1

# R1 — track association health (0.10 before the keyframe-to-keyframe fix;
# higher is better — every re-observation lengthens a smart-factor track):
grep -o "Smart factors: [0-9]* new, [0-9]* re-observed" /tmp/wslam_debug_run.log \
  | awk '{new+=$3; reob+=$5} END {print "re-observation rate:", reob/(new+reob)}'

# R2 — IMU-predicted step sanity (was spiking to ~17 m with the unit-norm
# warmup; want max < ~0.5 m for V101's gentle motion):
grep -o "metric_step=[0-9.]*" /tmp/wslam_debug_run.log \
  | sort -t= -k2 -n | tail -3

# R3 — gravity init: |g| ≈ 9.81, stationary_window_found=true, window
# timestamps inside the grounded lead-in, and gyro_bias ≈ V101's true bias
# (-0.002, 0.021, 0.078) rad/s:
grep "Initialised gravity" /tmp/wslam_debug_run.log

# Keyframe cadence + per-keyframe association quality:
grep -c "Accepted keyframe" /tmp/wslam_debug_run.log
grep "Accepted keyframe" /tmp/wslam_debug_run.log | tail -3   # reobserved=, median_parallax=

# Reference feature set behaviour (should advance only on acceptance after
# the bootstrap, and the first keyframe should anchor at motion onset):
grep -m1 "motion onset" /tmp/wslam_debug_run.log
grep -o "previous held for [0-9]* frames" /tmp/wslam_debug_run.log | sort | uniq -c
```

## 5. Visual check (worth 30 seconds, catches what stats hide)

```bash
cd benchmarks/results/$EXPERIMENT
../../.venv/bin/evo_traj tum wslam.tum --ref gt.tum -as \
    --save_plot wslam_vs_gt.png --plot_mode xyz
```

Open `*_trajectories.png` (shape overlay), `*_xyz.png` (per-axis vs time),
`*_speeds.png` (estimated speed profile vs GT — dead-reckoning failures show
up here first). The current failure mode looks like a small smooth blob in
the room centre; success looks like the ORB-SLAM3 plot (lines overlap).

## 6. Comparing two experiments directly

```bash
../../.venv/bin/evo_ape tum gt.tum wslam.tum -as \
    --save_results wslam_ape.zip          # inside each experiment dir
# then, from benchmarks/results/:
../.venv/bin/evo_res */wslam_ape.zip -p   # table + box plot across runs
```

## Pitfalls

- The binary needs `EUROC_DIR` and `WSLAM_SHADER_SRC_DIR`; pass them inline
  (values at the top of this file) or `source .env`.
- `benchmarks/results/` is gitignored — copy numbers/plots you want to keep
  into the paper notes, or commit the experiment dirs deliberately.
- compare.py's default `--out-dir benchmarks/results` overwrites the previous
  un-named run's `.tum` files; that's why step 2 names a subdirectory.
- evo associates poses by timestamp — if you change keyframe timing/export,
  conversions still work (the converter sorts and rescales automatically).
- A change in `--max-iters` changes how much of the sequence wslam covers;
  only compare full-sequence runs against full-sequence runs.
- ORB-SLAM3 is non-deterministic across runs (multi-threaded); its APE on
  V101 fluctuates a few mm around 0.04 m. Don't re-run it expecting identical
  numbers; re-running it is only needed when the dataset changes.
