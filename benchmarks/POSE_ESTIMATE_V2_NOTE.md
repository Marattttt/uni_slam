# Pose estimate v2: port, measurements and next steps

Measurements of the v2 pose estimate stage (`wslam/src/pose_estimate_v2/`,
RaCo-ALIKED-LightGlue matches → RANSAC → triangulation) against the v1 stage
(`wslam/src/pose_estimate_cpu/`, GPU ORB/BRIEF → CPU matcher → RANSAC →
triangulation) and EuRoC ground truth, on the keyframes v1 selected on
V1_01_easy.

## Setup

- Code: branch `feat/pose-estimate-v2` at `8f9f09241ce`. Debug build, AMD
  Radeon 610M (integrated), ONNX Runtime WebGPU EP.
- Model: `raco_aliked_lightglue_pipeline_k1024.onnx` (1024 keypoints per image).
- Keyframes: `wslam/tests/data/v1_keyframes_V101.json`, 50 frames (670–830,
  0-based cam0 indices), so **49 keyframe pairs**. Both pipelines read only
  those frames through `MakeKeyframeFilter`. v1 has no mapping stage here, so
  the test advances its reference feature set every frame, and both stages
  estimate exactly the previous keyframe → current keyframe motion.
- Ground truth: `state_groundtruth_estimate0`. The body pose is interpolated at
  each frame timestamp (lerp + slerp; both bracketing samples within 10 ms),
  `T_W_C = T_W_B · T_BS(cam0)`, and the truth for `p_curr = R p_prev + t` is
  `T_Ccurr_Cprev = T_W_Ccurr⁻¹ · T_W_Cprev`.
- Metrics:
  - rotation error: angle of `R_est · R_gtᵀ`
  - translation direction error: angle between the unit `t` and the GT
    translation, only for GT baselines ≥ 5 cm (22 pairs). Below that, the
    direction is noise for any two-view method.
- Reproduce: `source .env && WSLAM_POSE_REPORT_OUT=report.md
  ./build/wslam/tests/wslam_tests --gtest_filter=*Dataset*` (~3.6 min).

## Results

Median / p90 / max, degrees:

| | v1 before the LOD fix | v1 | v2 |
|---|---|---|---|
| poses recovered | 49/49 | 48–49/49 | **49/49** |
| rotation error vs GT | 1.89 / 2.76 / 3.56 | 0.83–0.92 / 1.70–1.89 / 3.30–3.52 | **0.20 / 0.36 / 0.59** |
| translation direction error vs GT (22 pairs) | 77.5 / 96.2 / 98.2 | 40.8–45.6 / 71.8–76.5 / 86.0–88.1 | **5.8 / 21.9 / 28.4** |
| v1 vs v2 rotation | 1.95 | 0.79–0.84 | – |

- "v1 before the LOD fix" is one run with `map::util::ToLod0Pixel` temporarily
  restored to ignore `Feature::lod` (commit `90314ef4fe8` fixes it).
- The v1 ranges cover two runs of the same code. **v1 is not deterministic:**
  the per-pair match counts differ between runs, most likely because the GPU
  feature arrays are filled through atomics. **v2 was bit-identical across all
  three runs.**
- Wall time, including setup: v1 ~32 s. v2 ~183 s, of which ~3.7 s per pair is
  the ORT model run; the three pose passes take ~0.1 s per pair.
- RANSAC (v2): inlier ratio 93.3–99.5%. The adaptive iteration count drew 5.3
  samples on average, 12 at most, and never reached the 10 000 guard. v1
  capped this at 500 iterations.
- The dataset test's thresholds were fixed before the first run, and all of
  them pass:
  - v2 recovers ≥ 90% of pairs
  - v2 median rotation error ≤ 0.5°
  - v2 median direction error ≤ 10°
  - v2 is not worse than v1
  - median v1↔v2 rotation ≤ 1°

  The last check has the thinnest margin (0.79–0.84°). That margin is set by
  v1's own error and run-to-run spread, not by v2. Against v1 before the LOD
  fix it fails (1.95°).

### Per pair (final run)

`dir err` is empty below a 5 cm baseline. Columns:
- `inl/matches`: RANSAC inliers / matches into RANSAC
- `lm`: landmarks surviving the depth and reprojection filters
- `reproj px`: their mean reprojection error

| # | frames | GT rot | GT baseline (m) | v1 rot err | v2 rot err | v1 dir err | v2 dir err | v1-v2 rot | v1-v2 dir | v1 inl/matches | v2 inl/matches | v1 lm | v2 lm | v1 reproj px | v2 reproj px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 670-671 | 1.893 | 0.026 | 0.627 | 0.122 | - | - | 0.747 | 75.340 | 1217/1624 | 891/904 | 569 | 890 | 0.490 | 0.226 |
| 1 | 671-672 | 1.962 | 0.028 | 0.128 | 0.078 | - | - | 0.066 | 42.480 | 1117/1419 | 883/892 | 1041 | 881 | 0.686 | 0.174 |
| 2 | 672-673 | 2.051 | 0.028 | 0.326 | 0.102 | - | - | 0.270 | 18.578 | 1310/1576 | 883/899 | 1285 | 881 | 0.889 | 0.176 |
| 3 | 673-674 | 2.115 | 0.029 | 0.468 | 0.024 | - | - | 0.471 | 61.565 | 1635/1878 | 891/896 | 1121 | 887 | 1.653 | 0.184 |
| 4 | 674-675 | 2.248 | 0.029 | 0.165 | 0.073 | - | - | 0.103 | 68.566 | 1293/1588 | 875/884 | 1220 | 872 | 1.568 | 0.202 |
| 5 | 675-676 | 2.142 | 0.029 | 0.484 | 0.068 | - | - | 0.445 | 20.963 | 1314/1652 | 871/887 | 1271 | 870 | 1.894 | 0.155 |
| 6 | 676-677 | 2.021 | 0.029 | 0.669 | 0.021 | - | - | 0.665 | 59.611 | 1146/1483 | 860/873 | 790 | 855 | 0.309 | 0.207 |
| 7 | 677-678 | 2.062 | 0.029 | 0.279 | 0.054 | - | - | 0.243 | 70.412 | 1216/1607 | 872/880 | 1066 | 869 | 0.532 | 0.193 |
| 8 | 678-680 | 3.888 | 0.059 | 1.035 | 0.070 | 8.495 | 5.685 | 0.996 | 5.052 | 1102/1456 | 848/859 | 1045 | 847 | 2.572 | 0.245 |
| 9 | 680-682 | 3.480 | 0.058 | 1.310 | 0.098 | 13.203 | 1.769 | 1.219 | 12.838 | 1062/1358 | 814/829 | 165 | 814 | 3.462 | 0.175 |
| 10 | 682-684 | 3.786 | 0.055 | 1.207 | 0.132 | 50.207 | 5.796 | 1.102 | 44.432 | 1081/1379 | 836/862 | 805 | 833 | 1.330 | 0.217 |
| 11 | 684-686 | 3.400 | 0.051 | 0.818 | 0.214 | 30.214 | 5.849 | 0.702 | 29.495 | 1169/1476 | 848/865 | 1135 | 842 | 0.699 | 0.245 |
| 12 | 686-688 | 3.051 | 0.048 | 1.240 | 0.233 | - | - | 1.189 | 27.904 | 1237/1561 | 865/870 | 463 | 861 | 3.449 | 0.446 |
| 13 | 688-690 | 3.005 | 0.047 | 1.022 | 0.287 | - | - | 0.786 | 32.829 | 1177/1441 | 862/872 | 1049 | 861 | 1.092 | 0.428 |
| 14 | 690-692 | 3.123 | 0.045 | 1.341 | 0.322 | - | - | 1.280 | 41.522 | 1449/1639 | 868/875 | 1029 | 862 | 3.004 | 0.653 |
| 15 | 692-694 | 3.099 | 0.042 | 0.906 | 0.135 | - | - | 0.783 | 55.114 | 1684/1818 | 864/878 | 1639 | 863 | 1.489 | 0.278 |
| 16 | 694-696 | 3.073 | 0.038 | 0.630 | 0.249 | - | - | 0.486 | 48.310 | 1533/1745 | 885/897 | 1451 | 884 | 0.702 | 0.335 |
| 17 | 696-698 | 2.842 | 0.034 | 0.880 | 0.274 | - | - | 0.688 | 23.387 | 1520/1732 | 879/888 | 1499 | 873 | 1.036 | 0.616 |
| 18 | 698-700 | 2.371 | 0.030 | 0.955 | 0.180 | - | - | 1.121 | 82.624 | 1576/1752 | 882/890 | 821 | 882 | 0.468 | 0.569 |
| 19 | 700-703 | 3.113 | 0.040 | 0.977 | 0.059 | - | - | 1.019 | 54.912 | 1530/1669 | 856/868 | 1475 | 856 | 0.753 | 0.179 |
| 20 | 703-717 | 4.630 | 0.092 | 0.294 | 0.588 | 41.633 | 28.367 | 0.315 | 13.712 | 1508/1655 | 878/888 | 1088 | 873 | 1.832 | 1.494 |
| 21 | 717-720 | 2.543 | 0.047 | 1.130 | 0.289 | - | - | 1.199 | 66.939 | 1462/1615 | 840/845 | 1265 | 819 | 0.944 | 0.716 |
| 22 | 720-724 | 2.289 | 0.065 | 1.554 | 0.315 | 35.624 | 21.995 | 1.745 | 57.011 | 1319/1436 | 804/817 | 1259 | 801 | 1.021 | 0.925 |
| 23 | 724-731 | 2.289 | 0.108 | 1.597 | 0.329 | 47.405 | 5.283 | 1.765 | 45.530 | 1367/1533 | 849/866 | 1032 | 849 | 1.697 | 0.378 |
| 24 | 731-737 | 3.254 | 0.092 | 3.154 | 0.185 | 67.517 | 5.740 | 3.066 | 72.256 | 1246/1394 | 855/859 | 955 | 855 | 1.290 | 0.264 |
| 25 | 737-739 | 1.598 | 0.040 | 0.392 | 0.144 | - | - | 0.266 | 41.199 | 1114/1222 | 857/866 | 1085 | 857 | 0.401 | 0.235 |
| 26 | 739-741 | 2.280 | 0.045 | 0.879 | 0.190 | - | - | 0.947 | 85.753 | 918/1036 | 866/873 | 531 | 866 | 0.664 | 0.347 |
| 27 | 741-743 | 2.265 | 0.046 | 0.985 | 0.348 | - | - | 1.144 | 93.371 | 1059/1176 | 867/879 | 520 | 867 | 0.797 | 0.601 |
| 28 | 743-745 | 2.249 | 0.046 | 0.465 | 0.247 | - | - | 0.318 | 22.153 | 986/1081 | 844/853 | 982 | 844 | 1.894 | 0.284 |
| 29 | 745-747 | 2.163 | 0.044 | 1.097 | 0.281 | - | - | 1.096 | 123.335 | 846/978 | 849/866 | 428 | 849 | 0.861 | 0.540 |
| 30 | 747-749 | 1.833 | 0.042 | 0.674 | 0.418 | - | - | 0.558 | 55.739 | 1010/1166 | 883/893 | 639 | 883 | 2.139 | 0.270 |
| 31 | 749-752 | 1.447 | 0.058 | 0.545 | 0.564 | 33.376 | 21.861 | 0.942 | 55.097 | 1108/1236 | 851/876 | 1096 | 851 | 1.155 | 0.898 |
| 32 | 752-757 | 2.161 | 0.085 | 0.872 | 0.110 | 24.419 | 11.224 | 0.910 | 24.528 | 1239/1366 | 877/888 | 1230 | 877 | 0.924 | 0.259 |
| 33 | 757-763 | 1.838 | 0.104 | 1.600 | 0.222 | 44.128 | 3.502 | 1.558 | 45.891 | 1188/1299 | 837/860 | 1023 | 837 | 0.998 | 0.221 |
| 34 | 763-766 | 1.570 | 0.064 | 1.699 | 0.222 | 84.354 | 13.668 | 1.643 | 92.782 | 1173/1314 | 859/877 | 586 | 859 | 2.149 | 0.368 |
| 35 | 766-769 | 0.689 | 0.068 | 0.231 | 0.214 | 23.980 | 11.898 | 0.300 | 22.037 | 1223/1323 | 869/885 | 1200 | 869 | 1.834 | 0.566 |
| 36 | 769-773 | 1.131 | 0.095 | 2.477 | 0.148 | 85.966 | 3.979 | 2.520 | 86.823 | 1021/1149 | 837/859 | 496 | 837 | 1.617 | 0.287 |
| 37 | 773-776 | 1.121 | 0.076 | 1.088 | 0.149 | 71.776 | 3.004 | 1.166 | 71.771 | 1046/1125 | 866/874 | 798 | 866 | 2.201 | 0.285 |
| 38 | 776-779 | 0.992 | 0.078 | 0.588 | 0.174 | 26.488 | 5.984 | 0.553 | 21.180 | 912/1000 | 819/833 | 896 | 818 | 1.030 | 0.380 |
| 39 | 779-783 | 0.975 | 0.103 | 1.836 | 0.063 | 69.967 | 4.165 | 1.875 | 66.263 | 640/767 | 767/783 | 319 | 766 | 2.033 | 0.221 |
| 40 | 783-787 | 0.997 | 0.105 | 0.601 | 0.129 | 15.529 | 3.896 | 0.670 | 14.650 | 482/569 | 699/723 | 465 | 699 | 1.985 | 0.201 |
| 41 | 787-791 | 2.428 | 0.104 | 0.485 | 0.371 | 70.379 | 8.227 | 0.823 | 62.531 | 306/365 | 669/712 | 218 | 669 | 1.146 | 0.247 |
| 42 | 791-795 | 2.562 | 0.085 | 0.520 | 0.201 | 60.651 | 10.193 | 0.679 | 50.473 | 201/267 | 716/742 | 162 | 716 | 1.067 | 0.322 |
| 43 | 795-812 | 2.912 | 0.186 | 3.295 | 0.339 | 70.472 | 3.622 | 3.615 | 66.925 | 180/240 | 659/704 | 138 | 659 | 0.829 | 0.266 |
| 44 | 812-819 | 4.122 | 0.060 | 0.834 | 0.182 | 47.043 | 3.642 | 0.904 | 49.041 | 162/200 | 680/709 | 141 | 680 | 1.720 | 0.366 |
| 45 | 819-821 | 2.787 | 0.013 | 0.521 | 0.277 | - | - | 0.276 | 86.228 | 155/186 | 731/751 | 114 | 544 | 0.629 | 0.430 |
| 46 | 821-824 | 3.769 | 0.021 | 0.796 | 0.362 | - | - | 0.500 | 110.404 | 193/213 | 692/738 | 145 | 453 | 2.208 | 0.598 |
| 47 | 824-826 | 1.855 | 0.018 | 0.308 | 0.257 | - | - | 0.098 | 40.675 | 202/228 | 741/761 | 148 | 642 | 0.464 | 0.456 |
| 48 | 826-830 | 2.231 | 0.044 | 0.522 | 0.336 | - | - | 0.204 | 24.001 | 136/162 | 713/764 | 101 | 713 | 1.588 | 0.495 |

## Findings

1. **v2 is far more accurate than v1 at the keyframe edge.** Rotation error is
   about 4× lower (0.20° vs 0.83–0.92° median), and translation direction is
   7–8× better (5.8° vs 41–46°). v1's translation direction is close to
   uninformative: a median of 41–46° and a p90 of 72–76°. Its rotation is
   usable but noisy.
2. **The v1 LOD bug roughly doubled v1's rotation error.** `ToLod0Pixel`
   ignored `Feature::lod`, so every feature from pyramid level ≥ 1 reached the
   triangulation's essential-matrix fit with coordinates shrunk by 1.2^lod.
   They had passed RANSAC, which scales correctly. Fixing it took v1 from 1.89°
   to 0.83–0.92° median rotation error. v1 mapping-stage accuracy has not
   been re-measured since the fix.
3. **v1's remaining error has visible sources:**
   - features above LOD 0 have integer coordinates on their own level, so they
     are quantised to 1.2^lod px at LOD 0
   - its F is fitted on distorted pixels
   - per-pair mean reprojection error: median 1.15 px, max 3.5 px, against
     0.29 px / 1.5 px for v2
   - match supply collapses from frame ~779 onwards (1400–1900 matches before,
     160–770 after), while v2 keeps 700+
4. **v2's worst pairs are the long or low-parallax gaps.** Pair 20 (703→717,
   14 frames) has a 0.59° rotation error, 28° direction error and 1.5 px
   reprojection. Pairs 22 and 31 have ~22° direction error at 5.8–6.5 cm
   baselines. These are linear eight-point estimates with no non-linear
   refinement, so a two-view refinement would act exactly here.
5. **Eight-point F RANSAC is fragile at low parallax (synthetic finding).**
   - Setup: 300 points, 0.3 m baseline at 2–8 m (~27 px median parallax),
     0.5 px noise, 25% gross outliers.
   - What happened: the best sample model accepted one gross outlier (moved
     ~320 px, but near that model's epipolar line). The least-squares re-fit
     with it is so weakly constrained that the worst Sampson distance of the
     true inliers rose from 1.6 to 57 px², and ~6% of them were lost (recall
     0.936).
   - Fitting E (5 DoF) on the same contaminated set is no better (max 75 px²).
     Without that one point, the re-fit recovers all 235 inliers.
   - At a 1.2 m baseline the problem disappears, which is why
     `RansacSeparatesInliersFromOutliers` uses that baseline (see its comment).
   - On the real pairs LightGlue's outlier rate is low (0.5–7%), so this did
     not show up in the dataset numbers. It is still a real weakness, inherited
     from v1's algorithm.
6. **v1 is slow on static scenes because of its SVDs.** During V101's
   stationary lead-in, v1 matches ~4500 features per frame. `RansacCPU`'s
   re-fit and `TriangulateCPU`'s essential fit then each run a `JacobiSVD`
   with `ComputeFullU` on an N×9 matrix, which builds an N×N U. That takes
   ~2.3 s + ~2.7 s per frame in Debug, so the first 100 frames of a v1 run
   take ~6 min. v2 computes V only. Separately, on those stationary frames F
   RANSAC reports 100% inliers after 1 iteration: a zero-motion pair is
   degenerate for F.

## Possible improvements

### Approach

- **Robust local optimisation in RANSAC** (LO-RANSAC / MSAC scoring /
  Huber-IRLS re-fit on the Sampson error), against finding 5. This matters
  more than the choice of minimal solver.
- ~~Essential-matrix RANSAC on the rays~~, ~~non-linear two-view
  refinement~~ and ~~LightGlue's match scores~~: done, see "Match scores and
  essential-matrix RANSAC" below. A minimal 5-point solver (Nistér) is still
  open, but at 93–99.5% inliers it would save samples, not accuracy.
- **F/H model selection** for low-parallax and pure-rotation pairs (ORB-SLAM
  style). The first 8 pairs are 50 ms apart with 2.6–2.9 cm baselines.
- **IMU:** a gyro-preintegrated rotation prior to validate or seed R, and to
  disambiguate degenerate pairs.
- **Triangulation:** optimal (Lindstrom) or midpoint instead of linear DLT, and
  a max rather than mean two-view reprojection gate.
- **Keypoint reuse across runs:** the left image of pair k+1 is the right image
  of pair k, but the model detects it again. Splitting detector and matcher
  (or batching pairs) would halve detection, the dominant cost at ~3.7 s per
  pair in Debug. It would also give keypoints a stable identity across runs,
  which landmark tracks need once v2 gets a mapping stage.
- **A v2 keyframe gate:** `PreservePyramidPass` copies Current → Previous every
  frame. In a keyframe-gated pipeline it must only copy on acceptance, the
  analogue of v1's `FeatureReferenceAdvanceName`.

### Implementation

- Move v1's RANSAC and triangulation onto `pose_estimate_v2/util`. That covers
  `ransac_cpu.cpp`, `triangulate_cpu.cpp` and keyframe_gate's own copies of
  `UndistortNormalised` and `RotationAngle`. It also fixes finding 6 for v1.
  `ransac_cpu.cpp` and `match_features_cpu.cpp` still carry their own (correct)
  LOD scaling instead of `map::util::ToLod0Pixel`.
- The v1 and v2 `SensorLoaderPass` are identical copies that now both carry
  the reading filter; merge them.
- `Compute::execute` turns `kComputeStopExecution` into success, so `main`
  loops until `--max-iters` even after the stream ended. The sensor loaders
  now stop cleanly at the end of the stream; `Compute` needs to surface that.
- Make v1 detection deterministic (sort each feature block after the atomic
  fill) so v1 numbers are reproducible.
- The v2 GUI is not ported.
- ~~The model's pixel-centre convention~~: checked against the graph, see
  below.
- The undistortion (`map::util::UndistortNormalised`) is a fixed-point
  iteration whose rate is about |2·k1·r²| (≈ 0.46 in EuRoC's corners). Twenty
  iterations are plenty, but Newton would converge in 3–4.

## Match scores and essential-matrix RANSAC

The sections above measure the stage as first ported. This section records
the follow-up that uses the model's extra outputs and drops the fundamental
matrix. Measured on the same 49 V101 keyframe pairs, RelWithDebInfo; v2 is
still bit-reproducible.

### What the model gives beyond matches (read from the ONNX graph)

- **Outputs:** `keypoints [2,1024,2]`, `matches [M,3]` and `mscores [M]`. This
  export has no covariance output: RaCo's covariance head is not in the graph.
- **`mscores` is already cut at > 0.1 inside the graph**, LightGlue's default
  `filter_threshold` (constant `val_4026`). Any floor below 0.1 is a no-op.
- **Keypoints come out of a sorted, descending `TopK`.** A keypoint's index is
  its RaCo rank. The RaCo paper uses ranking only to truncate a keypoint
  budget, so it has no two-view use. It may matter for tracks once v2 has
  mapping.
- **Pixel centre:** keypoints are `integer pixel + soft-argmax offset + 0.5`
  (constant `val_169 = 0.5`). The centre of the top-left pixel is therefore
  (0.5, 0.5), as `raco_lightglue.hpp` assumes, and `ToSourcePixel`'s −0.5 is
  correct. This closes the item in Implementation above.

### What other LightGlue systems do

| System | Score use | Geometric gate |
|---|---|---|
| LightGlue (default) | keep score > 0.1 | — |
| hloc `superpoint+lightglue` | `match_threshold` 0.2 | — |
| SuperVINS (README; VINS-Fusion + SP/LG) | `matche_score_threshold` 0.5 | F RANSAC `F_threshold: 1` px |
| DL-VINS-Factory (RaCo+LG on EuRoC, arXiv 2607.01757) | score ≥ β (value not given); RaCo covariance-weighted residuals, −9% median ATE | MAGSAC++ on **E**, 1 px epipolar |
| LightGlue authors (cvg/LightGlue#56) | — | OpenCV **E**-RANSAC, best at 1 px; PoseLib LO-RANSAC swept 0.5–3 px |
| RaCo paper (arXiv 2602.15755) | — | PoseLib RANSAC, threshold swept {0.5, …, 3.0} px |

These systems use score floors from 0.1 to 0.5. The calibrated systems all
estimate E. The confidence pay-off that is reported comes from weighting
residuals (DL-VINS's covariances), not from hard floors.

### What changed

- **E instead of F.**
  - The camera is calibrated and the rays are already undistorted, so RANSAC
    now fits eight-point **essential** matrices on the rays directly
    (`util::EstimateEssentialRansac`).
  - The Sampson inlier test is scaled to pixels by the mean focal length. The
    gate stays at 3.84 px², and the threshold was not swept.
  - The triangulation decomposes RANSAC's E instead of re-fitting it.
  - The final re-fit iterates (re-fit, re-classify; up to 3 rounds while the
    inlier set grows) and always returns the fit on the returned inliers.
- **Score floor** (`MatchFilterOptions`): drops matches below `min_score`. If
  fewer than 100 survive, it keeps the best-scoring 100.
- **Score weights** (`ScoreWeighting`): scale each constraint in the final
  linear E re-fit, and each residual in the refinement.
- **Non-linear refinement** (`util::RefineRelativePose`):
  - Levenberg-Marquardt on (R, t ∈ S²), 5 DoF.
  - Minimises the weighted Huber (1 px) Sampson error in pixels.
  - Numeric Jacobian.
  - Runs between the cheirality pick and the DLT.

### Ablation (`PoseEstimateDatasetTest.V2MatchScoreAblation`)

The test runs the model once and re-estimates every pair from the recorded
matches under each config. Rotation error is over all 49 pairs; direction
error is over the 22 pairs with a ≥ 5 cm GT baseline. Degrees:

| config | rot med | rot p90 | rot mean | dir med | dir p90 | dir mean | dir max |
|---|---|---|---|---|---|---|---|
| F pipeline as ported (before) | 0.201 | 0.362 | – | 5.77 | 21.86 | – | 28.37 |
| E, no floor, unweighted, no refinement | 0.192 | 0.421 | 0.220 | 5.64 | 21.33 | 9.29 | 26.75 |
| E + floor 0.2 | 0.203 | 0.551 | 0.243 | 5.53 | 23.69 | 9.24 | 30.75 |
| E + floor 0.5 | 0.192 | 0.376 | 0.236 | 6.63 | 32.27 | 11.33 | 38.38 |
| E + floor 0.9 | 0.142 | 0.429 | 0.201 | 7.02 | 20.13 | 10.02 | 25.99 |
| E + weights score | 0.152 | 0.418 | 0.203 | 5.99 | 16.62 | 8.61 | 25.79 |
| E + weights score² | 0.142 | 0.355 | 0.173 | 6.30 | 15.97 | 8.30 | 25.83 |
| E + refinement | 0.107 | 0.269 | 0.135 | 4.72 | 10.70 | 5.92 | 15.73 |
| E + weights score + refinement | 0.082 | 0.212 | 0.122 | 4.76 | 7.88 | 5.21 | 10.42 |
| E + floor 0.5 + refinement | 0.082 | 0.323 | 0.116 | 4.98 | 6.96 | 5.13 | 9.56 |
| **E + floor 0.5 + weights score + refinement (defaults)** | **0.082** | **0.215** | **0.106** | **4.63** | **6.98** | **4.94** | **7.65** |
| E + floor 0.5 + weights score² + refinement | 0.082 | 0.219 | 0.112 | 4.60 | 6.95 | 5.20 | 12.39 |

The full 22-row table is printed by the test. All configs recover 49/49
poses. The floor at 0.5 drops 5.3% of the 41 412 matches. RANSAC still keeps
97.9% of the rest.

### Findings

1. **F → E alone is a wash:** 0.192° vs 0.201° rotation, 5.64° vs 5.77°
   direction median. It is still the right model: it removes the redundant
   second fit and two spurious DoF. But the error was never in the choice of
   F vs E.
2. **Non-linear refinement is the single biggest gain.** It roughly halves
   rotation error (median 0.19° → 0.11°) and halves p90 direction error
   (21° → 11°). The linear eight-point fit minimises an algebraic error that
   over-weights some correspondences. That is exactly finding 4's weakness on
   long and low-parallax pairs, and it fixes those pairs:

   | pair | frames | dir err, F before | dir err, defaults |
   |---|---|---|---|
   | 20 | 703-717 | 28.4 | 4.4 |
   | 22 | 720-724 | 22.0 | 5.7 |
   | 31 | 749-752 | 21.9 | 3.4 |

3. **The scores help as weights, not as floors.**
   - Floors on their own move the numbers around without a trend: p90
     direction error goes 21 → 24 → 32 → 20 for floors 0 / 0.2 / 0.5 / 0.9.
     They remove few matches, since LightGlue's scores are mostly near 1, and
     those matches were mostly inliers anyway.
   - Score weights help in every setting. With refinement they take p90
     direction error from 10.7° to 7.9° and max from 15.7° to 10.4°.
   - With refinement and weights on, the 0.5 floor improves p90, mean and max
     direction error further. It does so under all three weightings. 0.5 is
     SuperVINS's value.
   - `score` and `score²` are within noise of each other. `score` is the
     default because its worst pair is better (7.65° vs 12.39°).
4. **Not everything improved.**
   - Pair 33 (757-763) went from 3.9° (E plain) to 7.0°.
   - Pair 23 (724-731) went from 5.3° (F) to 7.0°.
   - The defaults' errors are now concentrated in a 3–8° band instead of a
     long tail. With 22 scored pairs, differences under ~0.3° in the medians
     are noise.
5. **The defaults were chosen on the same 49 pairs they are measured on**, so
   treat the last row as an upper bound on what to expect elsewhere. The
   choice was deliberately coarse:
   - It is the simplest combination that beat the others on every direction
     summary.
   - The literature's 0.5 floor was used, not a tuned value.
   - The inlier gate was left at the conventional 1 px σ.

### Not done

- PROSAC or score-guided sampling. At ≥ 93% inliers RANSAC needs about 5
  samples, so ordering cannot help.
- Covariance weighting. The output isn't exported.
- Use of RaCo rank. There are no tracks yet.
- An inlier-threshold sweep. It was deliberately left out.
