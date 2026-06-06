#!/usr/bin/env python3
"""Convert SLAM trajectory outputs to TUM format and compare them with evo.

Inputs (all optional, but at least one trajectory is required):
  - wslam map.json   (`wslam::ExportMap` sidecar: keyframes with timestamp_ns,
                      R_world_cam 3x3, t_world_cam)
  - ORB-SLAM3 f_*.txt / kf_*.txt  (already TUM-ordered `t tx ty tz qx qy qz qw`
                      but with nanosecond timestamps, which evo's TUM parser
                      does not accept — detected and divided by 1e9)
  - EuRoC ground truth data.csv   (nanosecond timestamps, quaternion stored
                      w-first; reordered to TUM's x y z w)

Every input is written as a `.tum` file into --out-dir. Unless --convert-only
is given and a ground truth was provided, evo_ape and evo_rpe are then run for
each estimate against the ground truth (the exact commands are printed so they
can be re-run by hand).

Frame caveat: wslam poses are camera-frame, ORB-SLAM3's EuRoC save and the
ground truth are body/IMU-frame. The `-as` (Umeyama + scale) alignment absorbs
the global part of that difference; the constant camera<->body lever arm
(a few cm on EuRoC) remains a small systematic offset.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

NANOSECONDS_PER_SECOND = 1e9
# Timestamps this large cannot be seconds (would be year ~5138); treat as ns.
NANOSECOND_THRESHOLD = 1e14


def rotation_matrix_to_quaternion(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a quaternion in TUM order (x, y, z, w).

    Shepperd's method: pick the largest of the four squared components as the
    division pivot for numerical stability.
    """
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2][1] - m[1][2]) * s
        qy = (m[0][2] - m[2][0]) * s
        qz = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / norm, qy / norm, qz / norm, qw / norm


def write_tum(path: Path, rows: list[tuple[float, ...]]) -> None:
    """Write rows of (t, tx, ty, tz, qx, qy, qz, qw) sorted by timestamp."""
    rows.sort(key=lambda row: row[0])
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in rows:
            f.write(" ".join(f"{value:.9f}" for value in row) + "\n")
    print(f"wrote {len(rows):6d} poses -> {path}")


def convert_wslam_map_json(map_json_path: Path, out_path: Path) -> None:
    """wslam keyframes (R_world_cam / t_world_cam, ns timestamps) -> TUM."""
    with map_json_path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    format_version = int(document.get("format_version", -1))
    if format_version != 1:
        raise ValueError(
            f"{map_json_path}: unsupported format_version {format_version}"
        )

    rows = []
    for keyframe in document["keyframes"]:
        timestamp = int(keyframe["timestamp_ns"]) / NANOSECONDS_PER_SECOND
        tx, ty, tz = (float(v) for v in keyframe["t_world_cam"])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(keyframe["R_world_cam"])
        rows.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
    write_tum(out_path, rows)


def convert_orbslam_txt(trajectory_path: Path, out_path: Path) -> None:
    """ORB-SLAM3 EuRoC-style save (TUM-ordered, ns timestamps) -> TUM."""
    rows = []
    with trajectory_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [float(value) for value in line.split()]
            if len(fields) != 8:
                raise ValueError(
                    f"{trajectory_path}: expected 8 columns, got {len(fields)}"
                )
            timestamp = fields[0]
            if timestamp > NANOSECOND_THRESHOLD:
                timestamp /= NANOSECONDS_PER_SECOND
            rows.append((timestamp, *fields[1:]))
    write_tum(out_path, rows)


def convert_euroc_groundtruth_csv(csv_path: Path, out_path: Path) -> None:
    """EuRoC state_groundtruth_estimate0/data.csv -> TUM.

    Columns: timestamp[ns], p x y z, q w x y z, (velocity/bias columns ignored).
    """
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            timestamp = int(fields[0]) / NANOSECONDS_PER_SECOND
            tx, ty, tz = (float(v) for v in fields[1:4])
            qw, qx, qy, qz = (float(v) for v in fields[4:8])
            rows.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
    write_tum(out_path, rows)


def find_evo_binary(name: str) -> Path:
    """Locate an evo CLI tool next to this interpreter (the benchmarks venv)."""
    candidate = Path(sys.executable).parent / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"{name} not found at {candidate}; run this script with "
        "benchmarks/.venv/bin/python (pip install evo)"
    )


def run_evo_comparison(groundtruth_tum: Path, estimate_tum: Path) -> bool:
    """Run evo_ape and evo_rpe for one estimate; returns True on success."""
    ok = True
    commands = [
        [str(find_evo_binary("evo_ape")), "tum",
         str(groundtruth_tum), str(estimate_tum), "-as"],
        [str(find_evo_binary("evo_rpe")), "tum",
         str(groundtruth_tum), str(estimate_tum), "-as",
         "--delta", "1", "--delta_unit", "m"],
    ]
    for command in commands:
        print(f"\n$ {' '.join(command)}")
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"command failed with exit code {result.returncode}",
                  file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wslam", type=Path,
                        help="wslam map.json (ExportMap sidecar)")
    parser.add_argument("--orb", type=Path, action="append", default=[],
                        help="ORB-SLAM3 trajectory txt (repeatable)")
    parser.add_argument("--gt", type=Path,
                        help="EuRoC ground truth data.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="directory for the converted .tum files "
                             "(default: benchmarks/results)")
    parser.add_argument("--convert-only", action="store_true",
                        help="only write .tum files, skip the evo runs")
    args = parser.parse_args()

    if args.wslam is None and not args.orb and args.gt is None:
        parser.error("nothing to do: pass at least one of --wslam/--orb/--gt")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    estimates = []
    if args.wslam is not None:
        out_path = args.out_dir / "wslam.tum"
        convert_wslam_map_json(args.wslam, out_path)
        estimates.append(out_path)
    for orb_path in args.orb:
        out_path = args.out_dir / (orb_path.stem + ".tum")
        convert_orbslam_txt(orb_path, out_path)
        estimates.append(out_path)

    groundtruth_tum = None
    if args.gt is not None:
        groundtruth_tum = args.out_dir / "gt.tum"
        convert_euroc_groundtruth_csv(args.gt, groundtruth_tum)

    if args.convert_only:
        return 0
    if groundtruth_tum is None:
        print("\nno --gt given; skipping evo comparison")
        return 0
    if not estimates:
        print("\nno estimates given; skipping evo comparison")
        return 0

    all_ok = True
    for estimate_tum in estimates:
        print(f"\n=== {estimate_tum.name} vs ground truth ===")
        all_ok &= run_evo_comparison(groundtruth_tum, estimate_tum)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
