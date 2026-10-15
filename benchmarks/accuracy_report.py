#!/usr/bin/env python3
"""Accuracy report of TUM trajectories against a ground truth TUM file.

Complements compare.py's evo runs with the numbers a visual-inertial estimate
needs read together (see HOWTO_REEVALUATE.md for the reading order):

  coverage      share of the ground truth's time span the estimate covers,
                and how late it starts (a late bootstrap hides part of the
                flight from every other metric)
  scale         Umeyama Sim(3) scale; for a VIO it should be ~1
  APE           translation RMSE under Sim(3), SE(3) and, when the map json
                carries gravity, position + yaw (4-DoF, the VIO-observable
                gauge); rotation RMSE under SE(3), raw and with the constant
                body-frame offset to the ground truth removed (estimated, and
                reported: EuRoC's ground truth body frame differs from both
                ORB-SLAM3's and wslam's by a few degrees); RMSE per third of
                the run (drift shows as growth)
  RPE           over 1 / 5 / 10 m of ground-truth path: translation error as a
                share of the distance, metric (SE(3), what a VIO claims) and
                after the Sim(3) scale, plus rotation per metre; in the
                offset-corrected body frame (a body-frame offset rotates every
                relative translation and would read as RPE)
  local scale   Sim(3) scale of sliding windows: spread = scale drift
  gravity       angle between the estimate's gravity (map json) and the
                ground truth's in the first keyframe's camera frame; floored by
                the ground truth's own tilt against the accelerometer (2.75 deg
                on V1_01_easy, parked)
  time offset   APE-minimising shift (sanity check of the timestamps)
  loops         relative-position error of loop closure pairs after the
                Sim(3) alignment (map json with a loop_closure object)

Run standalone or through compare.py --report. Needs numpy and scipy (the
benchmarks venv has both).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Largest timestamp gap between an estimate pose and its ground truth match
MAX_ASSOCIATION_S = 0.01
RPE_DISTANCES_M = (1.0, 5.0, 10.0)
WINDOW_S = 5.0
WINDOW_STEP_S = 2.5
MIN_WINDOW_POSES = 8
TIME_OFFSETS_S = np.arange(-0.10, 0.1001, 0.01)
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Trajectory:
    stamps: np.ndarray       # [N] seconds
    positions: np.ndarray    # [N, 3]
    rotations: Rotation      # N rotations, world <- body

    @staticmethod
    def load_tum(path: Path) -> "Trajectory":
        data = np.loadtxt(path, ndmin=2)
        return Trajectory(data[:, 0], data[:, 1:4],
                          Rotation.from_quat(data[:, 4:8]))

    def subset(self, mask: np.ndarray) -> "Trajectory":
        return Trajectory(self.stamps[mask], self.positions[mask],
                          self.rotations[np.flatnonzero(mask)])


@dataclass
class EstimateInputs:
    label: str
    tum: Path
    map_json: Path | None = None


@dataclass
class Alignment:
    scale: float
    rotation: np.ndarray     # 3x3
    translation: np.ndarray  # 3

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.rotation.T + self.translation


@dataclass
class Report:
    label: str
    lines: list[str] = field(default_factory=list)

    def add(self, name: str, value: str) -> None:
        self.lines.append(f"| {name} | {value} |")


def umeyama(source: np.ndarray, target: np.ndarray,
            with_scale: bool) -> Alignment:
    """Least-squares (s, R, t) with target ~ s R source + t."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    scale = 1.0
    if with_scale:
        variance = (source_centered ** 2).sum() / len(source)
        scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return Alignment(scale, rotation, translation)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    cosine = float(np.dot(a, b))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        # Half turn about any axis perpendicular to a
        perpendicular = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(
            np.pi * perpendicular / np.linalg.norm(perpendicular)).as_matrix()
    return Rotation.from_rotvec(axis / sine * np.arctan2(sine, cosine)).as_matrix()


def align_position_yaw(source: np.ndarray, target: np.ndarray,
                       gravity_rotation: np.ndarray) -> Alignment:
    """4-DoF alignment: gravity_rotation levels the source (its gravity onto
    -z), then the yaw about z and the translation are fitted."""
    leveled = source @ gravity_rotation.T
    source_mean = leveled.mean(axis=0)
    target_mean = target.mean(axis=0)
    s = leveled - source_mean
    t = target - target_mean
    yaw = np.arctan2((s[:, 0] * t[:, 1] - s[:, 1] * t[:, 0]).sum(),
                     (s[:, 0] * t[:, 0] + s[:, 1] * t[:, 1]).sum())
    yaw_rotation = Rotation.from_rotvec(yaw * UP).as_matrix()
    rotation = yaw_rotation @ gravity_rotation
    translation = target_mean - yaw_rotation @ source_mean
    return Alignment(1.0, rotation, translation)


def associate(estimate: Trajectory, truth: Trajectory,
              offset_s: float = 0.0) -> tuple[Trajectory, Trajectory]:
    """Nearest ground-truth pose per estimate pose; unmatched ones dropped."""
    stamps = estimate.stamps + offset_s
    index = np.clip(np.searchsorted(truth.stamps, stamps), 1,
                    len(truth.stamps) - 1)
    earlier = np.abs(truth.stamps[index - 1] - stamps) \
        < np.abs(truth.stamps[index] - stamps)
    index = np.where(earlier, index - 1, index)
    matched = np.abs(truth.stamps[index] - stamps) <= MAX_ASSOCIATION_S
    truth_matched = Trajectory(truth.stamps[index[matched]],
                               truth.positions[index[matched]],
                               truth.rotations[index[matched]])
    return estimate.subset(matched), truth_matched


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


def ape(estimate: Trajectory, truth: Trajectory,
        alignment: Alignment) -> np.ndarray:
    return np.linalg.norm(alignment.apply(estimate.positions)
                          - truth.positions, axis=1)


def rotation_ape_deg(estimate: Trajectory, truth: Trajectory,
                     alignment: Alignment) -> np.ndarray:
    aligned = Rotation.from_matrix(alignment.rotation) * estimate.rotations
    return np.degrees((truth.rotations.inv() * aligned).magnitude())


def body_offset(estimate: Trajectory, truth: Trajectory,
                alignment: Alignment) -> Rotation:
    """Constant rotation X with R_truth ~ R_align R_estimate X^-1: a fixed
    difference between the estimate's and the ground truth's body frames
    (an extrinsic mismatch), which the world alignment cannot absorb. The
    chordal mean of the per-pose differences, projected onto SO(3)"""
    aligned = Rotation.from_matrix(alignment.rotation) * estimate.rotations
    differences = (truth.rotations.inv() * aligned).as_matrix()
    u, _, vt = np.linalg.svd(differences.sum(axis=0))
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1.0
    return Rotation.from_matrix(u @ sign @ vt)


def with_body_offset(estimate: Trajectory, offset: Rotation) -> Trajectory:
    """The estimate with its body frame turned onto the ground truth's"""
    return Trajectory(estimate.stamps, estimate.positions,
                      estimate.rotations * offset.inv())


def path_lengths(positions: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def rpe(estimate: Trajectory, truth: Trajectory, distance_m: float,
        scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative errors over pairs `distance_m` of ground-truth path apart:
    translation error (m, estimate scaled by `scale`), rotation error (deg)
    and the pairs' ground-truth distance."""
    along = path_lengths(truth.positions)
    ends = np.searchsorted(along, along + distance_m)
    starts = np.flatnonzero(ends < len(along))
    ends = ends[starts]
    if len(starts) == 0:
        empty = np.array([])
        return empty, empty, empty

    def relative(trajectory: Trajectory, factor: float):
        inverse = trajectory.rotations[starts].inv()
        delta = (trajectory.positions[ends] - trajectory.positions[starts])
        return (inverse.apply(delta) * factor,
                inverse * trajectory.rotations[ends])

    truth_translation, truth_rotation = relative(truth, 1.0)
    estimate_translation, estimate_rotation = relative(estimate, scale)
    translation_error = np.linalg.norm(estimate_translation
                                       - truth_translation, axis=1)
    rotation_error = np.degrees(
        (truth_rotation.inv() * estimate_rotation).magnitude())
    return translation_error, rotation_error, along[ends] - along[starts]


def window_scales(estimate: Trajectory, truth: Trajectory) -> np.ndarray:
    scales = []
    start = estimate.stamps[0]
    while start + WINDOW_S <= estimate.stamps[-1]:
        mask = (estimate.stamps >= start) & (estimate.stamps < start + WINDOW_S)
        if mask.sum() >= MIN_WINDOW_POSES:
            scales.append(umeyama(estimate.positions[mask],
                                  truth.positions[mask], True).scale)
        start += WINDOW_STEP_S
    return np.array(scales)


def read_map_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stats_cell(values: np.ndarray, unit: str, digits: int = 3) -> str:
    if len(values) == 0:
        return "n/a"
    return (f"{rmse(values):.{digits}f} {unit} RMSE (mean {np.mean(values):.{digits}f}, "
            f"median {np.median(values):.{digits}f}, max {np.max(values):.{digits}f})")


def evaluate(inputs: EstimateInputs, truth_full: Trajectory,
             cam_in_body: np.ndarray | None) -> Report:
    report = Report(inputs.label)
    document = read_map_json(inputs.map_json)
    estimate_full = Trajectory.load_tum(inputs.tum)
    estimate, truth = associate(estimate_full, truth_full)
    if len(estimate.stamps) < 3:
        report.add("error", "fewer than 3 poses match the ground truth")
        return report

    # Coverage
    truth_span = truth_full.stamps[-1] - truth_full.stamps[0]
    covered = estimate.stamps[-1] - estimate.stamps[0]
    report.add("poses (matched / total)",
               f"{len(estimate.stamps)} / {len(estimate_full.stamps)}")
    report.add("coverage of the ground truth's time span",
               f"{100.0 * covered / truth_span:.1f}% ({covered:.1f} of "
               f"{truth_span:.1f} s); starts {estimate.stamps[0] - truth_full.stamps[0]:.1f} s "
               f"after the ground truth")
    truth_path = path_lengths(truth.positions)[-1]
    report.add("ground-truth path covered", f"{truth_path:.2f} m")

    # Alignments
    sim3 = umeyama(estimate.positions, truth.positions, True)
    se3 = umeyama(estimate.positions, truth.positions, False)
    report.add("**Umeyama scale** (Sim(3))",
               f"{sim3.scale:.4f} ({100.0 * (sim3.scale - 1.0):+.2f}%)")
    estimated_path = path_lengths(estimate.positions)[-1]
    report.add("path length estimate / truth",
               f"{estimated_path / truth_path:.4f}")

    sim3_errors = ape(estimate, truth, sim3)
    report.add("APE translation, Sim(3)", stats_cell(sim3_errors, "m"))
    report.add("APE translation, SE(3)",
               stats_cell(ape(estimate, truth, se3), "m"))

    gravity = document.get("gravity_world")
    if gravity is not None:
        gravity_direction = np.asarray(gravity, dtype=float)
        gravity_direction /= np.linalg.norm(gravity_direction)
        leveling = rotation_between(gravity_direction, -UP)
        posyaw = align_position_yaw(estimate.positions, truth.positions,
                                    leveling)
        report.add("APE translation, position + yaw (4-DoF)",
                   stats_cell(ape(estimate, truth, posyaw), "m"))
        if cam_in_body is not None:
            # Ground-truth gravity in the first keyframe's camera frame (the
            # estimate's world frame)
            first_body = truth.rotations[0].as_matrix()
            camera_in_world = first_body @ cam_in_body[:3, :3]
            truth_gravity = camera_in_world.T @ -UP
            angle = np.degrees(np.arccos(np.clip(
                np.dot(truth_gravity, gravity_direction), -1.0, 1.0)))
            report.add("gravity direction error at the first keyframe",
                       f"{angle:.2f} deg (against the ground truth's "
                       f"orientation, which is itself 2.75 deg off a parked "
                       f"accelerometer's gravity on V1_01_easy)")
    else:
        report.add("APE translation, position + yaw (4-DoF)",
                   "n/a (map json has no gravity_world)")

    report.add("APE rotation, SE(3)",
               stats_cell(rotation_ape_deg(estimate, truth, se3), "deg", 2))
    offset = body_offset(estimate, truth, se3)
    estimate_body = with_body_offset(estimate, offset)
    report.add("constant body-frame offset to the ground truth",
               f"{np.degrees(offset.magnitude()):.2f} deg (removed below: "
               f"rotation APE and RPE use the offset-corrected body frame)")
    report.add("APE rotation, SE(3), offset removed",
               stats_cell(rotation_ape_deg(estimate_body, truth, se3), "deg",
                          2))

    thirds = np.array_split(np.arange(len(sim3_errors)), 3)
    report.add("APE Sim(3) RMSE by third of the run",
               " / ".join(f"{rmse(sim3_errors[part]):.3f}" for part in thirds)
               + " m")

    for distance in RPE_DISTANCES_M:
        metric_t, metric_r, spans = rpe(estimate_body, truth, distance, 1.0)
        if len(metric_t) == 0:
            report.add(f"RPE over {distance:g} m", "n/a (path too short)")
            continue
        scaled_t, _, _ = rpe(estimate_body, truth, distance, sim3.scale)
        report.add(f"RPE over {distance:g} m",
                   f"translation {rmse(metric_t):.3f} m RMSE "
                   f"({100.0 * np.median(metric_t / spans):.1f}% median) metric, "
                   f"{rmse(scaled_t):.3f} m ({100.0 * np.median(scaled_t / spans):.1f}%) "
                   f"after Sim(3) scale; rotation {rmse(metric_r):.2f} deg RMSE "
                   f"({np.median(metric_r / spans):.3f} deg/m median)")

    scales = window_scales(estimate, truth)
    if len(scales):
        p5, p50, p95 = np.percentile(scales, [5, 50, 95])
        report.add(f"local scale ({WINDOW_S:g} s Sim(3) windows)",
                   f"P5 {p5:.3f}, median {p50:.3f}, P95 {p95:.3f} "
                   f"(spread {100.0 * (p95 - p5):.1f}%)")

    offsets = []
    for offset in TIME_OFFSETS_S:
        shifted, shifted_truth = associate(estimate_full, truth_full, offset)
        alignment = umeyama(shifted.positions, shifted_truth.positions, True)
        offsets.append(rmse(ape(shifted, shifted_truth, alignment)))
    best = int(np.argmin(offsets))
    report.add("APE-minimising time offset",
               f"{TIME_OFFSETS_S[best] * 1000:+.0f} ms "
               f"(APE {offsets[best]:.3f} m vs {rmse(sim3_errors):.3f} m at 0)")

    loops = document.get("loop_closure", {}).get("loops", [])
    if loops:
        stamp_of = {int(kf["id"]): int(kf["timestamp_ns"]) / 1e9
                    for kf in document.get("keyframes", [])}

        def index_of(keyframe: int) -> int | None:
            stamp = stamp_of.get(keyframe)
            if stamp is None:
                return None
            i = int(np.argmin(np.abs(estimate.stamps - stamp)))
            return i if abs(estimate.stamps[i] - stamp) < 1e-4 else None

        aligned = sim3.apply(estimate.positions)
        residuals = []
        for loop in loops:
            query = index_of(int(loop["query"]))
            candidate = index_of(int(loop["candidate"]))
            if query is None or candidate is None:
                continue
            residuals.append(np.linalg.norm(
                (aligned[query] - aligned[candidate])
                - (truth.positions[query] - truth.positions[candidate])))
        if residuals:
            report.add(f"loop pairs: relative position error ({len(residuals)})",
                       f"mean {np.mean(residuals):.3f} m, median "
                       f"{np.median(residuals):.3f} m")

    return report


def render(reports: list[Report]) -> str:
    out = []
    for report in reports:
        out.append(f"### {report.label}\n")
        out.append("| metric | value |")
        out.append("|---|---|")
        out.extend(report.lines)
        out.append("")
    return "\n".join(out)


def build_report(groundtruth_tum: Path, estimates: list[EstimateInputs],
                 cam_in_body: np.ndarray | None = None) -> str:
    truth = Trajectory.load_tum(groundtruth_tum)
    return render([evaluate(estimate, truth, cam_in_body)
                   for estimate in estimates])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, required=True,
                        help="ground truth .tum (compare.py writes gt.tum)")
    parser.add_argument("--est", type=Path, action="append", required=True,
                        help="estimate .tum (repeatable)")
    parser.add_argument("--map-json", type=Path, action="append", default=[],
                        help="map json of the --est at the same position "
                             "(gravity and loops), repeatable")
    parser.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args()

    estimates = []
    for i, tum in enumerate(args.est):
        map_json = args.map_json[i] if i < len(args.map_json) else None
        estimates.append(EstimateInputs(tum.stem, tum, map_json))
    text = build_report(args.gt, estimates)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
