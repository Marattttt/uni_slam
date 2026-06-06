# wslam accuracy analysis — EuRoC V1_01_easy vs ORB-SLAM3 (2026-06-04)

This document records the first quantitative accuracy evaluation of `pc_wslam`
against ORB-SLAM3 (mono-inertial) and EuRoC ground truth, explains the
methodology and the GTSAM machinery involved (assuming no prior GTSAM
knowledge), and derives — with code- and log-level evidence — why the current
error is high and what to change. Everything here is reproducible with the
commands in the appendix.

---

## 1. TL;DR

| Metric (Sim(3)-aligned vs GT) | wslam | ORB-SLAM3 mono-inertial |
|---|---|---|
| APE RMSE (absolute trajectory error) | **1.62 m** | **0.036 m** |
| RPE RMSE per 1 m travelled | 2.76 m | 0.095 m |
| Raw path length (GT: 58.6 m) | 51.6 m ✓ | 58.0 m ✓ |
| Umeyama scale correction | **0.18** ⚠ | ~1.0 |

The headline 1.62 m is **not** a scale problem and **not** an export/convention
bug. The trajectory's total metric length is right (IMU integration works on
average), but its *shape* barely correlates with the ground truth — the best
Sim(3) fit shrinks it 5.5× into a blob. Three concrete, fixable defects were
identified and confirmed empirically:

1. **Frame-vs-keyframe geometry conflation** — the back-end attaches
   *consecutive-frame* measurements to *keyframe-to-keyframe* graph edges
   (keyframes are ~3–5 frames apart). Confirmed: only **10.3 %** of landmark
   observations continue an existing track; ~90 % of landmarks live for
   exactly one keyframe pair.
2. **Unit-norm translation warmup** — the first 10 keyframes feed the
   optimiser ~1 m odometry steps when the true motion is millimetres.
   Confirmed: IMU-predicted step lengths later spike to **17.3 m per 0.2 s**
   (86 m/s indoors) — the velocity/bias state never recovers.
3. **Gravity initialised mid-flight** — the 200-frame cold-start skip places
   the "stationary" gravity window at t≈283.3 s, *after* the MAV takes off,
   and gravity is never refined afterwards.

---

## 2. Benchmark setup

- Sequence: EuRoC `V1_01_easy` (`EUROC_DIR` from `.env`), ~144 s, 58.6 m
  ground-truth path, room-scale (~10 × 10 × 3 m) with multiple loops.
- Ground truth: `state_groundtruth_estimate0/data.csv` (200 Hz, body/IMU
  frame, nanosecond timestamps, quaternion stored w-first).
- Baseline: ORB-SLAM3 (upstream `master` @ `4452a3c4ab`, vendored under
  `benchmarks/ORB_SLAM3/` via git subtree), mono-inertial example, default
  EuRoC config. Full-frame trajectory `f_dataset-V101_monoi.txt`.
- wslam: `map.json` exported by `wslam::ExportMap` (585 keyframes over the
  132 s the run covered).
- Tooling: `benchmarks/compare.py` converts all three sources to TUM format
  (`timestamp_sec tx ty tz qx qy qz qw`); [evo](https://github.com/MichaelGrupp/evo)
  computes APE/RPE. Plots live in `benchmarks/results/*.png`.

**Known methodological caveats** (small, do not change conclusions):

- wslam exports camera-frame poses; ORB-SLAM3's EuRoC save and the GT are
  body/IMU-frame. The constant camera↔body lever arm (a few cm on EuRoC) is a
  small systematic offset that Sim(3) alignment mostly absorbs.
- wslam's trajectory is keyframes-only (585 poses) vs ORB-SLAM3's full 2660;
  evo associates by timestamp, so sparser sampling slightly *flatters* RPE.
- ORB-SLAM3 has loop closure + full bundle adjustment; on a room-scale looped
  sequence this effectively removes drift. 0.036 m matches its published V101
  result, which validates our build/run — it is a ceiling, not a fair
  same-class comparison for an in-development front-end.

---

## 3. How to read the evo numbers

- **TUM format**: one pose per line, `t x y z qx qy qz qw`, timestamp in
  seconds. evo associates estimate↔GT poses by nearest timestamp.
- **APE (absolute pose error)**: align the whole estimated trajectory to GT
  with one rigid (SE(3), flag `-a`) or similarity (Sim(3), flag `-as`)
  transform, then measure per-pose position error. Sensitive to accumulated
  drift.
- **RPE (relative pose error)**: compare relative motions over a fixed delta
  (here 1 m of travelled distance) between estimate and GT. Insensitive to
  global drift; measures *local* odometry quality.
- **Umeyama alignment**: the closed-form least-squares fit of
  rotation+translation(+scale) between two point sets. The fitted **scale**
  is itself diagnostic: for a metrically-correct VI trajectory it should be
  ≈1.0.

### The diagnosis chain from the numbers alone

| Evidence | Value | Inference |
|---|---|---|
| Raw path length | 51.6 m vs 58.6 m GT | metric scale source (IMU) is right *on average* |
| Umeyama scale | 0.18 | the fit prefers shrinking the trajectory 5.5× toward the centroid → its **shape is nearly uncorrelated** with GT (for a well-shaped but drifting trajectory this would stay ≈1) |
| APE SE(3)-only | 2.55 m (vs 1.62 m Sim(3)) | confirms shape, not scale, dominates |
| RPE per metre | 2.2 m median | even **local** relative motion is mostly wrong in direction |
| Aligned plots | wslam = smooth slow blob in the room centre; GT = large loops | classic heading-drift + lost-structure signature; the trajectory behaves like damped dead reckoning |

---

## 4. GTSAM primer, scoped to what wslam uses

Skip if you know GTSAM. Everything below maps 1:1 to code in `wslam/`.

**Factor graph**: a bipartite graph of *variables* (things to estimate) and
*factors* (probabilistic constraints between variables). Solving = finding
variable values that maximise the product of all factor likelihoods
(equivalently minimise the sum of weighted squared residuals). wslam's
variables, keyed per accepted keyframe `i` (`mapping_state.hpp:51-67`):

- `x_i` — `gtsam::Pose3`, camera-in-world pose (rotation + translation).
- `v_i` — `gtsam::Vector3`, velocity in world frame.
- `b_i` — `gtsam::imuBias::ConstantBias`, 6-DoF accel+gyro bias.

Landmarks are deliberately **not** variables (see smart factors below).

**Noise models / sigmas**: every factor carries a covariance. A small sigma =
"trust this measurement a lot". Units matter: a `BetweenFactor` translation
sigma of 1.0 m says "the measured relative translation may be off by ~1 m".
Factors with wrong measurements *and* confident sigmas actively pull the
solution toward the wrong answer.

**The factors wslam builds** (`factor_builder.cpp::execute`):

- `PriorFactor<Pose3>` on `x_0` (σ=1e-4) — pins the *gauge*: a SLAM graph is
  otherwise free-floating (any rigid transform of everything is an equally
  good solution).
- Priors on `v_0` (σ=0.1 m/s, "starts roughly stationary") and `b_0`.
- `CombinedImuFactor(x_i, v_i, x_j, v_j, b_i, b_j)` — the standard
  *preintegration* factor: all IMU samples between keyframes i and j are
  integrated into one relative motion constraint (rotation from gyro,
  velocity/position change from accel, gravity subtracted using a **fixed**
  `n_gravity` vector set once at init, `factor_builder.cpp:130`), plus a
  bias random-walk model. This is what makes monocular scale observable.
- `BetweenFactor<Pose3>(x_i, x_j)` — vision-only odometry from the
  essential-matrix decomposition (σ_rot=0.1 rad, σ_t=1.0 m). Its stated
  purpose (`factor_builder.cpp:305-315`) is conditioning iSAM2's landmark
  cliques, not metric truth.
- `SmartProjectionPoseFactor<Cal3_S2>` — one per landmark. Stores the pixel
  observations + observing pose keys; at every linearisation it internally
  triangulates the 3D point from the current pose estimates and marginalises
  it out. Consequence worth internalising: **a smart factor with only two
  observations whose poses are nearly collinear (or whose measurements are
  inconsistent) contributes almost nothing** — the internal triangulation is
  degenerate and `ZERO_ON_DEGENERACY` silently zeroes the factor.

**iSAM2** (`isam2_worker.cpp`): incremental smoothing — instead of
re-optimising the whole graph each keyframe, it maintains a Bayes tree and
only re-eliminates/relinearises the affected parts. wslam runs it on a worker
thread, one update in flight (`isam2_update.cpp`), with QR factorisation and
relinearisation every update for numerical robustness.

**Smart-factor lifecycle in wslam**: re-observing a landmark means cloning its
smart factor, appending the measurement, removing the old factor by index and
re-adding the clone (`factor_builder.cpp:441-479`). This remove-and-readd is
the standard pattern; it is bookkeeping-heavy but correct here.

---

## 5. Root causes (code-referenced, evidence-backed)

### R1 — Frame-vs-keyframe geometry conflation ★ the big one

The front-end matcher/RANSAC/triangulation always operate on **consecutive
frames** N-1 ↔ N (`LoadDataCPUPass::shiftFeatureSets` keeps exactly 2 frame
feature sets; `featuesets_stored = 2`, `common.hpp:50`). The keyframe gate,
however, accepts only ~1 frame in 3–5, and the back-end graph edge connects
**keyframe M → keyframe N**. Three distinct corruptions follow:

**R1a. The BetweenFactor measurement spans one frame, the edge spans many.**
`keyframe_gate.cpp:258-259` copies `tri.R_prev_to_curr / t_prev_to_curr`
(frame N-1 → N motion) into the delta; `factor_builder.cpp:316-327` attaches
it between `x_M` and `x_N`. With a mean gap of ~3–5 frames, the measured
rotation is ~⅓–⅕ of the true inter-keyframe rotation, every single edge, with
a fairly confident σ_rot = 0.1 rad. The graph is systematically told "you
barely rotated" — heading drift is the direct consequence, and heading drift
is exactly what the aligned plots show.

**R1b. The first observation of every landmark is attributed to the wrong
pose.** For a new landmark, `keyframe_gate.cpp:430-436` emits the `feat_prev`
pixel (measured in frame N-1) as an observation *from keyframe M's pose*
(M ≠ N-1 whenever the gap > 1). The smart factor then triangulates from two
rays, one of which originates from a camera position that may be several cm
to tens of cm away from where the pixel was actually observed. Every new
landmark is born geometrically corrupted.

**R1c. Track association is broken across gaps.** Landmark continuity is
keyed on exact `Feature` equality (`operator<=>` default — bit-exact x, y,
strength, orientation, descriptor; `models.hpp:109`) between the *previous
accepted keyframe's* feature set (`MappingState::active_landmarks`,
`keyframe_gate.cpp:388`) and `feat_prev` from frame N-1. These are only the
same objects when **two consecutive frames are both accepted** as keyframes.

> **Empirical confirmation** (debug run, 600 iterations, 191 keyframes):
> 11 236 new landmarks vs 1 295 re-observations → **10.3 % re-observation
> rate**. ~90 % of all smart factors hold exactly 2 observations (one of them
> mis-attributed per R1b) and are then abandoned. The map has essentially no
> medium-term structure to anchor the trajectory; the novelty gate
> (`min_new_landmarks`) is also vacuous since nearly everything always looks
> new.

### R2 — Unit-norm translation warmup poisons the bootstrap

The essential-matrix translation is unit-norm by construction
(`models.hpp:182-183`). `factor_builder.cpp:292-303` only rescales it to the
IMU-predicted metric step **after the 10th keyframe**. The first 10
BetweenFactors therefore claim ~1 m translation steps (σ_t = 1.0 m, so ~1σ
violations at minimum) while the true inter-keyframe motion at sequence start
is millimetres-to-centimetres. The optimiser reconciles this by bending the
velocity and bias states; those feed `pim.predict`, whose step length then
scales all *subsequent* between-translations.

> **Empirical confirmation**: per-keyframe IMU-predicted step lengths
> (`metric_step` debug log) over the run: median 0.027 m (plausible) but
> spikes of 0.41 m, 8.67 m, and a max of **17.3 m within a ~0.2 s window**
> (86 m/s for an indoor MAV). The velocity/bias subsystem is repeatedly
> driven into absurd states and the between-translation magnitudes inherit
> the garbage. Note the feedback loop: bad steps → bad poses → smart-factor
> triangulations rejected by `ZERO_ON_DEGENERACY` + the 5 px dynamic outlier
> threshold → even less vision to correct the IMU chain.

### R3 — Gravity initialised mid-flight, then frozen

`sensor_loader.hpp` skips the first **200 frames** (10 s) as cold-start. The
first accepted keyframe lands at t≈283.3 s — but V1_01_easy takes off around
t≈277 (visible in the GT speed plot). The "stationary window" gravity
estimate (`keyframe_gate.cpp:297-330`) therefore averages **20 IMU samples
(0.1 s) taken during flight**. The magnitude looks deceptively fine
(|g| = 9.821 m/s² logged, because hover thrust ≈ g on average), but the
*direction* carries the flight acceleration of that instant — and a tilt of
even 2° leaks g·sin(2°) ≈ 0.34 m/s² of phantom horizontal acceleration into
every preintegration window. Because `n_gravity` is set once into the
preintegration params (`factor_builder.cpp:130`) and there is no gravity
refinement anywhere in the graph, this error is permanent and, since the MAV
rotates constantly, cannot be absorbed by the accel bias either.
(ORB-SLAM3, for contrast, runs dedicated inertial-only optimisation stages —
the "VIBA 1/2" lines in its log — that refine gravity direction and biases.)

### R4 — Structural ceiling (expected at this stage, listed for the paper)

No loop closure, no relocalisation, no local BA window beyond what iSAM2's
incremental updates provide, 2-view-dominated landmarks (consequence of R1c).
Even with R1–R3 fixed, drift will accumulate without these; they explain the
gap to ORB-SLAM3's 0.036 m, not the 1.62 m blob.

---

## 6. Improvement plan (ranked by expected impact ÷ effort)

1. **Make the mapping path operate keyframe-to-keyframe (fixes R1a+b+c at
   once).** Keep per-frame matching for the gate signals if convenient, but
   once a frame is accepted as a keyframe, compute matches / RANSAC /
   triangulation **against the previous keyframe's feature set** (store it in
   `MappingState`), and build the delta from that. Then: the between-rotation
   measurement actually spans the edge, `feat_prev` observations belong to the
   pose they're attributed to, and `active_landmarks` association works at
   every keyframe instead of 10 % of them — tracks extend, smart factors
   accumulate 3+ observations, vision starts genuinely constraining the graph.
   - Cheaper interim variant: compose the per-frame `R_rel` across skipped
     frames (rotation composition is scale-free and exact) and drop the
     `feat_prev` observation unless the gap is exactly 1.
2. **Kill the unit-norm warmup (R2).** Never feed an unscaled essential-matrix
   translation into a metric factor. During the warmup window, either (a) use
   the IMU step from keyframe 1 (it is noisy but not 1000× wrong), or (b) emit
   a rotation-only between (huge translation sigma), letting the
   CombinedImuFactor own translation from the start. Also add a sanity clamp
   on `metric_step` (e.g. > 1 m per inter-keyframe window on EuRoC ⇒ log +
   fall back to previous step length) as a tripwire.
3. **Initialise gravity from a genuinely stationary window (R3).** Reduce
   `kInitialFramesToSkip` (200 → ~20) so the run starts pre-takeoff, and/or
   detect stationarity (gyro norm < threshold over ≥ 1 s of samples) before
   accepting the window. Longer term: refine gravity direction in the graph
   (2-DoF gravity variable or an ORB-SLAM3-style inertial-only optimisation
   after ~10 keyframes).
4. **Diagnostics to keep honest** (cheap, do alongside 1–3): log per-keyframe
   re-observation rate, frame gap, `metric_step`, and between-rotation vs
   gyro-rotation residual. Re-run `benchmarks/compare.py` after each fix —
   expected progression: Umeyama scale → ~1.0 first (shape recovers), then
   APE drops.
5. **Later (R4):** match new keyframes against the last N keyframes (not just
   1) to extend track lifetimes further; local BA window; loop closure.

---

## 7. What is defensible in a paper at the current stage

- **Architecture claims**: GPU compute front-end (pyramid, FAST-style
  detection, BRIEF descriptors) with batched single-submission stages; CPU
  geometric back-end; asynchronous iSAM2 smoothing on a worker thread with
  smart projection factors and combined IMU preintegration. All of this runs
  end-to-end in real time on the full sequence — that is demonstrated.
- **Methodology claims**: reproducible evaluation harness (TUM conversion +
  evo APE/RPE vs EuRoC GT) with ORB-SLAM3 as a pinned, version-tracked
  baseline whose measured accuracy (0.036 m APE on V101) matches its
  published results — i.e. the harness itself is validated.
- **Honest current-state numbers**: APE RMSE 1.62 m / Sim(3); the diagnosis
  that the error is shape- not scale-dominated (raw length 51.6/58.6 m,
  Umeyama scale 0.18); and the identified causes above with their empirical
  evidence (10.3 % re-observation rate, 17 m metric-step spikes, mid-flight
  gravity window). A failure analysis of this depth is presentable: it shows
  the system is instrumented, understood, and has a concrete path forward.
- **Do not claim**: competitive accuracy, loop-closure capability, or robust
  VI initialisation — yet.

---

## 8. Reproduction appendix

```bash
# 0. One-time setup (already done in this repo)
./benchmarks/build_orbslam3.sh
python -m venv benchmarks/.venv && benchmarks/.venv/bin/pip install evo numpy

# 1. Baseline trajectory (writes benchmarks/results/f_dataset-V101_monoi.txt)
./benchmarks/run_orbslam3_euroc.sh

# 2. wslam trajectory (writes map.ply + map.json in CWD)
./run.sh --map-out=map.ply          # or ./present_and_run.sh

# 3. Convert + compare (writes .tum files + prints APE/RPE)
source .env
benchmarks/.venv/bin/python benchmarks/compare.py \
    --wslam map.json \
    --orb benchmarks/results/f_dataset-V101_monoi.txt \
    --gt "$EUROC_DIR/state_groundtruth_estimate0/data.csv"

# 4. Diagnostics used in section 5
#    scale-vs-shape:
benchmarks/.venv/bin/evo_ape tum benchmarks/results/gt.tum \
    benchmarks/results/wslam.tum -a -v     # SE(3)
benchmarks/.venv/bin/evo_ape tum benchmarks/results/gt.tum \
    benchmarks/results/wslam.tum -as -v    # Sim(3); prints scale correction
#    plots (needs evo_config set plot_backend Agg once):
benchmarks/.venv/bin/evo_traj tum benchmarks/results/wslam.tum \
    --ref benchmarks/results/gt.tum -as --save_plot out.png --plot_mode xyz
#    instrumented wslam run (re-observation rate, gravity, metric_step):
source .env && SPDLOG_LEVEL=debug ./build/pc_wslam --max-iters=600 \
    > /tmp/wslam_debug_run.log 2>&1
grep "Initialised gravity" /tmp/wslam_debug_run.log
grep -o "Smart factors: [0-9]* new, [0-9]* re-observed" /tmp/wslam_debug_run.log \
  | awk '{new+=$3; reob+=$5} END {print reob/(new+reob)}'
grep -o "metric_step=[0-9.]*" /tmp/wslam_debug_run.log | sort -t= -k2 -n | tail -3
```

Artifacts referenced: `benchmarks/results/{wslam,gt,f_dataset-V101_monoi}.tum`,
`benchmarks/results/{wslam,orb}_vs_gt_{trajectories,xyz,speeds,rpy}.png`.
