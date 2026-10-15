# NOTE: v2 loop closure

What was built, why it looks the way it does, what was measured, and what is
still open. Code: `wslam/src/map_v2/loop_closure/` plus the glue in `map_v2/`
(`feed_loop_closure.*`, `map_stage.cpp`). Architecture summary for future
sessions is in `CLAUDE.md` (v2 mapping section, **Loop closure**).

## TL;DR

<!-- RESULTS -->

## What runs where

```
main thread (mapping stage, accepted keyframes only)
  Filter -> Drain -> Associate -> Build -> Submit -> FeedLoopClosure
                                                       | push (never waits)
                                                       v
loop closure thread (LoopClosureWorker, event queue)
  KeyframeImageEvent : PlaceImage + tracked landmark pixels
      -> describe (MixVPR | MegaLoc)            every keyframe
  MapUpdateEvent     : drained pose, archived poses + landmarks, gravity
      -> PoseGraph::addKeyframe (NodeGate)       node or attached
      -> PlaceDatabase::add (FAISS)              nodes only
  newest node of the batch -> query:
      FAISS top-N (id <= query - min_keyframe_gap)
      -> similarity / time-gap gates
      -> RaCo-LightGlue on (candidate, query)
      -> candidate keypoint -> landmark (nearest tracked pixel, 1 px)
         -> archived 3D point -> candidate camera frame
      -> PnP (DLT-6 RANSAC + Huber GN)
      -> consistency vs the graph's estimate
      -> PoseGraph::addLoop -> DriftCorrection mailbox
                                                       | latestCorrection()
main thread <------------------------------------------+
  SharedBindings::drift_correction (4-DoF, for live output)

finalize(): smoother flush -> final MapUpdate -> worker.finish()
            -> ApplyCorrections: map.json (corrected) + map_vio.json (VIO)
```

## Decisions (agreed in the architecture review)

1. **The smoother is never modified.** Its marginal factors are linearised in
   the VIO frame, so shifting the window (or adding a prior on the newest pose)
   would fight them. The loop closure publishes a VINS-Mono-style 4-DoF drift
   transform `T_drift = T_corr * T_vio^-1` (yaw about gravity + translation) and
   the export merges the pose graph into the map. Side effect worth knowing:
   **one run gives both trajectories** — `map_vio.json` is exactly what a
   `-no-loop-closure` run would produce (same keyframes, same smoother).
2. **"Meaningfully different pose" = node sparsification** (`NodeGate`,
   10 cm / 5 deg from the last node). A gated-out keyframe is attached to the
   last node: no FAISS entry, no query, its image is dropped, and at export it
   moves with that node.
3. **PnP correspondences come from the front end's RaCo-LightGlue model**,
   run on the (candidate, query) pair in a second session on the loop thread.
   The global descriptors carry no local features and the front end exports no
   per-keypoint descriptors, so there was nothing else to match with.
   Candidate keypoints are looked up by **pixel** (nearest tracked keypoint
   within 1 px), not by index, so a different EP (GPU front end, CPU matcher)
   that reorders TopK ties cannot break the association.
4. **Backlog: embed every keyframe, query only the newest node** of each batch.
5. **Roll/pitch held to the VIO** by a `Pose3AttitudeFactor` per node.
6. **GPU-mode MegaLoc runs on an auxiliary `wgpu::Device`** (same adapter, ORT
   WebGPU context 2), not the pipeline's: one device must not be driven by two
   threads' ORT sessions.
7. **No OpenCV** (changed mid-implementation, on your question): PnP is an own
   6-point DLT RANSAC with adaptive sample count + Huber Gauss-Newton on
   `gtsam::Pose3` (GTSAM's `PinholeBase::project2` Jacobians), and the squash
   resize is an own exact area-average resampler. Nothing unpinned enters the
   link line; FAISS is the only new dependency.

## Implementation process and pitfalls

- **Base**: the worktree was stale at `d54f59d`; fast-forwarded to `main`
  (`e8eaf2b`). `map_v2` was already merged — `worktree-feat+port-mapping-stage`'s
  tip `4e94411` is an ancestor of `main`, nothing to merge.
- **FAISS v1.15.1** (released 2026-09-16) via `FetchContent` from the release
  tarball, `URL_HASH SHA256=7b583010…0ffa`. Its defaults turn on GPU, MKL,
  Python, extras and Metal-on-Apple — all forced off; `FAISS_OPT_LEVEL=generic`
  builds the single `faiss` target (one descriptor per keyframe: SIMD buys
  nothing). It links the system reference `libblas`/`liblapack` and GCC's
  OpenMP. Built cleanly on the first try.
- **Model contracts** (checked at load, `ReadVprModelSpec`): input `image`
  float32 `[1,3,H,W]`, output `descriptor` float32 `[1,D]`, already
  L2-normalised; **neither export contains the ImageNet normalisation**, so
  `ToVprInput` does it. MegaLoc accepts 322x224 or 322x322 (W x H) and the size
  is read from the model, so switching sizes is only a path change.
- **Preprocessing parity**: the C++ MixVPR descriptor matches a numpy
  exact-area-resample + onnxruntime 1.31 reference at **cosine 0.99999994**.
  A first comparison against PIL's `Image.BOX` gave 0.988 — PIL's BOX is *not*
  exact area averaging (up to 41 gray levels off); the C++ side was right.
  MixVPR is input-sensitive: +-0.5 gray-level noise alone moves it to 0.994.
- **Robust kernel on loop factors was a bug, now opt-in** (`loop_robust_k = 0`).
  A true loop corrects drift that is many sigmas from the odometry chain; a
  Cauchy kernel at k = 1 sigma weights such a factor ~1/290 and the graph barely
  moves. Verification (PnP inliers + consistency bound) is the outlier gate.
- **Sigmas from measurements, not guesses**: odometry 0.005 rad / 2 cm per edge
  (the v2 smoother's measured RPE is 0.18 deg / 1.1 cm per keyframe on V101),
  loop 0.005 rad / 3 cm. With the first guesses (0.01 / 0.02 odometry,
  0.02 / 0.05 loop) the synthetic loop test removed exactly the 91% of yaw drift
  the sigma ratio predicts — correct behaviour, wrong weighting.
- **`compare.py` gained `--cam-to-body`** (EuRoC `cam0/sensor.yaml` T_BS) and a
  repeatable `--wslam`. Validated on the golden 1200-frame map: APE 0.149 m,
  matching the earlier hand-converted number.

<!-- MORE -->
