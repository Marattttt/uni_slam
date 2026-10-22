# NOTE: UZH-FPV indoor_45_2 — wslam vs ORB-SLAM3

Measured 2026-10-16 on `main` (665902f) plus the UZH-FPV support in this change. The
sequence is the one `NOTES_harder-dataset-comparison.md` lists under "Difficult".

## TL;DR

- **ORB-SLAM3 mono-inertial fails this sequence: 6 of 6 runs.**
  - Two configurations, 3 runs each: TUM-VI's fisheye settings (1250 features, FAST
    20/7) and a tuned one (2000 features, FAST 10/5).
  - Every run resets its map 48–66 times and ends with an empty atlas, so there is no
    trajectory to score.
  - Its maps live 0.6–0.9 s on average and 8.3 s at most.
- **wslam completes it in one map, with no resets.**
  - Global BA map: **0.051 m** APE Sim(3), 0.091 m SE(3), over 207.6 m of ground-truth
    path, with 96 % coverage.
  - Scale +0.96 %, RPE 1.2 % over 1 m, world-frame gravity error 0.17°.
- **The camera setup was checked against vision-only ORB-SLAM3** (it tracks), so the
  inertial failure is not a calibration mistake:
  - ORB-SLAM3 monocular (no IMU) on the same settings holds a 360-keyframe map:
    0.26 m Sim(3) APE (median 0.09 m) over 83 % of the ground truth.
  - Its scale is arbitrary (×3.7), so it is a sanity check, not a competitor.
- **The ground truth runs ~16 ms ahead of the IMU clock.** Score with
  `--gt-time-offset -0.016`. Without it, every wslam map reads 30–60 % worse (global BA
  0.082 m instead of 0.051 m).

## Dataset and setup

| | |
|---|---|
| Sequence | UZH-FPV `indoor_45_2`, Snapdragon text release (`indoor_45_2_snapdragon_with_gt.zip`, v3) + `indoor_45_calib_snapdragon.zip` |
| Sensors | stereo 640×480 equidistant fisheye (f ≈ 275 px; corners ~83° off-axis), 30 Hz; IMU 500 Hz; camera tilted 45° |
| Motion | parked 6 s, then a 69 s racing flight; ground truth covers 49.4 s / 207.6 m of it (1416.6–1466.0 s) |
| Calibration | Kalibr: `T_cam_imu`, `timeshift_cam_imu` = −14.849 ms, IMU noise 0.05 / 4e-5 (gyro), 0.1 / 0.002 (accel) — the same values for both systems |
| Machine | 16 cores / 32 threads, no discrete GPU; `--device=cpu` |
| wslam | `run_wslam_uzh_fpv.sh` — v2, mapping + loop closure (MixVPR), Release build at 640×480 (`build/release_640x480`), cam0 only |
| ORB-SLAM3 | `uzh_fpv_to_euroc.py` mirror + `run_orbslam3_uzh_fpv.sh` — `mono_inertial_euroc` (unpatched), `KannalaBrandt8`, cam0 only |
| Scoring | `compare.py --report --gt-time-offset -0.016 --cam-to-body <camchain>`; IMU (body) poses on the IMU clock |

Data lives in `~/dev/proj/resources/res_uni_slam/uzh_fpv/`:
- `indoor_45_2_snapdragon_with_gt/` — the sequence.
- `calib/indoor_45_calib_snapdragon/` — the calibration.
- `indoor_45_2_euroc/` — ORB-SLAM3's mirror and settings.
- `downloads/` — the zips, plus `indoor_45_2.zip`, the raw Leica log.

## Robustness and cost

| | maps / resets | result | wall time |
|---|---|---|---|
| wslam | 1 / 0 | 844 keyframes, 247 loops accepted, global BA accepted | 881 s (global BA 203 s) |
| ORB-SLAM3 mono-inertial, TUM-VI settings, r1 / r2 / r3 | 59 / 66 / 55 resets | empty atlas, no trajectory | real time |
| ORB-SLAM3 mono-inertial, 2000 features + FAST 10/5, r1 / r2 / r3 | 48 / 52 / 57 resets | empty atlas, no trajectory | real time |
| ORB-SLAM3 monocular (no IMU), TUM-VI settings, 1 run | 5 maps | last map 360 keyframes, 83 % of the ground truth | real time |

Why ORB-SLAM3 mono-inertial fails:
- **Inlier rule.** Before its IMU is initialised, it needs ≥ 50 local-map inliers on every
  frame (`Tracking.cc:3039`; vision-only needs 30, `:3057`).
- **Reset rule.** Losing track before IMU init resets the active map (`:2151`).
- **The scene.** The flight starts at liftoff over a low-texture concrete floor, and the
  fisheye sees mostly floor. Maps rarely survive the seconds the IMU initialisation needs.
- **Even past init.** The tuned runs reached the first inertial BA (`VIBA 1`) once, then
  lost that map as well.

wslam on this sequence:
- **Bootstrap.** It initialises gravity from the parked lead-in (score 0.033, 0.4 s
  window), and the first keyframe lands at liftoff (1404.6 s).
- **Model use.** The frame prefilter skipped 165 parked frames; the model ran on every
  flight frame (no motion-predicted skips).
- **Keyframe rate.** 844 keyframes over 69 s of flight is 12 per second, against ~5 per
  second on EuRoC.

## Accuracy (wslam)

Ground-truth clock corrected (`--gt-time-offset -0.016`; see below):

| metric | VIO (`map_vio`) | pose graph (`map_pgo`) | **global BA (`map`)** |
|---|---|---|---|
| Umeyama scale | +2.44 % | +2.81 % | +0.96 % |
| APE Sim(3) RMSE | 0.917 m | 0.093 m | **0.051 m** |
| APE SE(3) RMSE | 0.936 m | 0.235 m | 0.091 m |
| APE Sim(3) max | 1.886 m | 0.280 m | 0.080 m |
| gravity error (world) | 0.78° | 0.86° | 0.17° |
| rotation APE (offset removed) | 6.38° | 0.67° | 0.60° |
| RPE 1 m (median) | 3.1 % | 3.5 % | 1.2 % |
| RPE 5 m (median) | 2.6 % | 2.6 % | 1.2 % |
| RPE 10 m (median) | 2.6 % | 2.4 % | 1.3 % |
| local scale spread (5 s) | 4.0 % | 2.4 % | 1.5 % |
| loop pairs: relative position error (247) | 0.75 m median | 0.073 m | 0.011 m |

Reading it:
- **The VIO drifts most in yaw.** 6.4° rotation APE and 0.9 m APE over 207 m (0.44 %
  of the path).
  - Its local accuracy is better than on EuRoC: RPE 3 % and scale spread 4 %, against
    10–18 % on V1_03 / V2_03 (`NOTE_HARD_SEQUENCES.md`).
  - The fast, well-excited flight observes scale well.
- **The flight loops over the same arena constantly.** 247 accepted loops let the pose
  graph remove the drift (0.093 m).
- **The global BA halves that again.** Like on EuRoC, it is the largest single
  improvement in local error.
- **Coverage.** 703 of 844 keyframes fall inside the ground truth's window. The rest are
  the take-off and the landing approach, before and after it.

Raw clock (no `--gt-time-offset`), for comparison with other evaluations:
- VIO 0.920 m, pose graph 0.126 m, global BA 0.082 m.
- Each map's APE-minimising offset is +20 to +30 ms there.

## Timestamps: the camera shift is right, the ground truth is offset

The first scoring showed every map improving with a +20–30 ms shift. That is close to
twice Kalibr's 14.8 ms camera shift, so either the shift's sign or the ground truth's
clock was suspect.

1. **Reversed shift.** A wslam run with the opposite shift (`UZH_FPV_CAMERA_TIME_SHIFT_S
   =+0.014849`, `results/uzh_indoor_45_2/wslam_shift_reversed/`) scored better on the raw
   clock: global BA 0.068 m vs 0.082 m.
   - Its global BA cost per observation was the same (0.192 vs 0.193), so the cost
     can't tell the two runs apart.
2. **`benchmarks/gyro_time_offset.py`.** It needs no ground truth and no extrinsics: it
   compares a trajectory's rotation angles over 3-pose windows with the gyro integrated
   over the same windows, shifted by a lag.

   | trajectory | best lag | RMS at best / at 0 |
   |---|---|---|
   | ORB-SLAM3 monocular — vision only, camera stamps moved by Kalibr's shift | **0 ms** | 3.23 / 3.23 mrad (6.3 at ±15 ms) |
   | wslam global BA | +2 ms | 1.82 / 1.95 mrad |
   | `groundtruth.txt` | **−16 ms** | 0.32 / 0.40 mrad |

   - **Kalibr's shift (`t_imu = t_cam + shift`) puts the camera on the IMU clock.** The
     provider applies it correctly.
   - **The ground truth is on neither clock.** Its pose at t matches the gyro at
     t − 16 ms, about the raw camera clock.
3. **Corrected clock.** With the ground truth moved onto the IMU clock, the calibrated
   run wins on every map, and its residual offset is ~0 ms. The reversed run's residual
   is −20 to −30 ms: its fusion really is mis-timed.

   | APE Sim(3), clock corrected | VIO | pose graph | global BA |
   |---|---|---|---|
   | Kalibr shift (−14.8 ms) | **0.917 m** | **0.093 m** | **0.051 m** |
   | reversed (+14.8 ms) | 0.975 m | 0.125 m | 0.104 m |

## Fisheye handling in wslam

- **Distortion model.** v2 now undistorts with the equidistant model (Kalibr's
  Kannala-Brandt) and works on the normalised plane, as before.
- **Wide angles.** Rays near the image corners are ~83° off-axis (|ray| ≈ 8), where a
  pixel of noise is up to ~30× larger in pinhole units.
  - The Sampson gate (scaled by the focal length) and the smart factors (isotropic pixel
    noise) therefore weight the edges wrongly: the gate rejects good edge matches, and
    the factors trust them too much.
  - This run used no limit (`WSLAM_MAX_INCIDENCE_DEG`=90). Capping it is the obvious
    experiment; the knob exists, and the cap was not tried.
- **v1 refuses non-radial-tangential cameras.**

## Caveats

- **One sequence, and ORB-SLAM3 has no successful run.** This shows ORB-SLAM3
  mono-inertial is not robust on this sequence. It says nothing about accuracy on it.
  - ORB-SLAM3 stereo-inertial would likely do better. The release has both cameras, and
    the provider exposes both.
  - wslam uses cam0 only, so the mono-inertial comparison is the matching one.
- **Which poses are scored.** wslam is scored on keyframes. ORB-SLAM3's every-frame
  trajectory would have been scored, had it produced one.
- **Concurrent load.** ORB-SLAM3 inertial run 2 overlapped the vision-only check, and
  run 1 overlapped a short incremental compile. The failure mode is identical in all
  six runs.
- **IMU windows count samples.** The v2 IMU windows are counted in samples, tuned at
  EuRoC's 200 Hz: gravity 200, prefilter stationarity 50. At 500 Hz they span 0.4 s /
  0.1 s; the stationary scores here (0.03) are far below the 0.5 threshold, so it did not
  matter.

## Reproduce

```bash
U=~/dev/proj/resources/res_uni_slam/uzh_fpv
SEQ=$U/indoor_45_2_snapdragon_with_gt
CALIB=$U/calib/indoor_45_calib_snapdragon
cmake -S . -B build/release_640x480 -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DWSLAM_FRAME_WIDTH=640 -DWSLAM_FRAME_HEIGHT=480
cmake --build build/release_640x480 -j 16

./benchmarks/run_wslam_uzh_fpv.sh $SEQ $CALIB benchmarks/results/uzh_indoor_45_2/wslam

benchmarks/.venv/bin/python -I benchmarks/uzh_fpv_to_euroc.py $SEQ $CALIB $U/indoor_45_2_euroc
benchmarks/.venv/bin/python -I benchmarks/uzh_fpv_to_euroc.py $SEQ $CALIB $U/indoor_45_2_euroc \
    --orb-features 2000 --orb-fast 10 5 --settings-name orbslam3_monoi_2000f.yaml
for i in 1 2 3; do
    ./benchmarks/run_orbslam3_uzh_fpv.sh $U/indoor_45_2_euroc uzh-indoor_45_2 $i
    ORB_SETTINGS=orbslam3_monoi_2000f.yaml \
        ./benchmarks/run_orbslam3_uzh_fpv.sh $U/indoor_45_2_euroc uzh-indoor_45_2_2000f $i
done

W=benchmarks/results/uzh_indoor_45_2/wslam
benchmarks/.venv/bin/python benchmarks/compare.py --report --gt-time-offset -0.016 \
    --wslam $W/map_vio.json --wslam $W/map_pgo.json --wslam $W/map.json \
    --gt $SEQ/groundtruth.txt \
    --cam-to-body "$CALIB/camchain-imucam-..indoor_45_calib_snapdragon_imu.yaml" \
    --out-dir $W/eval_gt_clock

benchmarks/.venv/bin/python benchmarks/gyro_time_offset.py $SEQ/imu.txt \
    groundtruth=$SEQ/groundtruth.txt wslam=$W/eval_gt_clock/map.tum
```

`benchmarks/results/uzh_indoor_45_2/` (gitignored) holds:
- `wslam/`, with `eval/` (raw clock) and `eval_gt_clock/`.
- `wslam_shift_reversed/`.
- `orbslam3/`: the six inertial logs and the monocular run with its `eval_mono/`.
