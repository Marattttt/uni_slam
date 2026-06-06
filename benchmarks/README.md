# Benchmarks: wslam vs ORB-SLAM3 on EuRoC

Compares `pc_wslam` pose estimates against ORB-SLAM3 (mono-inertial) on the
EuRoC sequence configured in the repo's `.env` (`EUROC_DIR`, currently
V1_01_easy), using ground truth + the [evo](https://github.com/MichaelGrupp/evo)
toolset.

`ORB_SLAM3/` is vendored via `git subtree` (squashed) from
`https://github.com/UZ-SLAMLab/ORB_SLAM3.git` so the exact benchmarked version
is tracked. Local patches on top of upstream `master`:
- main `CMakeLists.txt`: C++17 (required by Pangolin 0.9 headers)
- `include/LoopClosing.h`: `mnFullBAIdx` `bool` → `int` (it is a generation
  counter that gets `++`-incremented; bool increment was removed in C++17)

## Workflow

```bash
# 1. One-time: build ORB-SLAM3 (reuses vendor/pangolin/install, system OpenCV)
./benchmarks/build_orbslam3.sh

# 2. One-time: venv for the comparison tooling
python -m venv benchmarks/.venv && benchmarks/.venv/bin/pip install evo numpy

# 3. Run ORB-SLAM3 mono-inertial on the .env EuRoC sequence
#    -> benchmarks/results/f_dataset-V101_monoi.txt (full trajectory)
#    -> benchmarks/results/kf_dataset-V101_monoi.txt (keyframes only)
#    Optional MAX_FRAMES arg limits the run to the first N images by
#    truncating the timestamps file the example iterates over (output is
#    then suffixed _<N>f) — used for fast 20%-segment iteration, see
#    HOWTO_REEVALUATE.md.
./benchmarks/run_orbslam3_euroc.sh          # full
./benchmarks/run_orbslam3_euroc.sh 582      # first 582 frames (~20%)

# 4. Produce a wslam map (writes map.ply + map.json)
./run.sh --map-out=map.ply        # or ./present_and_run.sh

# 5. Convert everything to TUM format and run evo APE/RPE vs ground truth.
#    compare.py needs no env vars — only the --gt path, which lives under
#    the dataset root (EUROC_DIR in .env):
benchmarks/.venv/bin/python benchmarks/compare.py \
    --wslam map.json \
    --orb benchmarks/results/f_dataset-V101_monoi.txt \
    --gt "<EUROC_DIR>/state_groundtruth_estimate0/data.csv"
```

`compare.py --convert-only` just writes the `.tum` files (into
`benchmarks/results/` by default) without running evo, e.g. for use with
`evo_traj tum --plot` by hand.

- Step-by-step re-evaluation checklist (per-experiment dirs, diagnostic
  greps, what "better" looks like): **`HOWTO_REEVALUATE.md`**
- Current numbers, root-cause analysis, GTSAM primer: **`ACCURACY_ANALYSIS.md`**

## Format notes

- TUM format: `timestamp_sec tx ty tz qx qy qz qw`, one pose per line.
- ORB-SLAM3's `SaveTrajectoryEuRoC` output is already TUM-ordered but uses
  **nanosecond** timestamps; `compare.py` auto-detects this and rescales.
- EuRoC ground truth stores quaternions **w-first**; `compare.py` reorders.
- Frames: wslam exports camera-frame poses, ORB-SLAM3's EuRoC save and the
  ground truth are body/IMU-frame. `evo_ape/_rpe -as` (Umeyama + scale)
  absorbs the global difference; the constant camera↔body lever arm (a few
  cm on EuRoC) remains a small systematic offset.
