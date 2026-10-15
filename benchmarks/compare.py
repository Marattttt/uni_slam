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
ground truth are body/IMU-frame. Pass --cam-to-body with the sequence's
cam0/sensor.yaml to convert wslam poses to the body frame first
(T_world_body = T_world_cam * T_BS^-1); without it the `-as` (Umeyama +
scale) alignment absorbs the global part of the difference, but RPE compares
camera-frame motion with body-frame motion and the camera<->body lever arm
(a few cm on EuRoC) remains a systematic offset.

--wslam is repeatable (e.g. a loop-corrected map.json and its map_vio.json);
each estimate is written as <stem>.tum.
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


def read_euroc_t_bs(sensor_yaml: Path) -> list[list[float]]:
    """T_BS (sensor in body) of a EuRoC sensor.yaml, as a 4x4 row-major list."""
    text = sensor_yaml.read_text(encoding="utf-8")
    start = text.index("T_BS")
    data_start = text.index("[", text.index("data:", start)) + 1
    data_end = text.index("]", data_start)
    values = [float(v) for v in text[data_start:data_end].replace("\n", " ").split(",")]
    if len(values) != 16:
        raise ValueError(f"{sensor_yaml}: T_BS has {len(values)} values")
    return [values[row * 4:(row + 1) * 4] for row in range(4)]


def invert_rigid(t: list[list[float]]) -> list[list[float]]:
    rotation_t = [[t[col][row] for col in range(3)] for row in range(3)]
    translation = [-sum(rotation_t[row][k] * t[k][3] for k in range(3))
                   for row in range(3)]
    return [rotation_t[row] + [translation[row]] for row in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
            for r in range(4)]


def convert_wslam_map_json(map_json_path: Path, out_path: Path,
                           cam_in_body: list[list[float]] | None = None) -> None:
    """wslam keyframes (R_world_cam / t_world_cam, ns timestamps) -> TUM.

    With cam_in_body (T_BS), poses are converted to the body frame."""
    body_in_cam = invert_rigid(cam_in_body) if cam_in_body is not None else None
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
        rotation = [[float(v) for v in row] for row in keyframe["R_world_cam"]]
        translation = [float(v) for v in keyframe["t_world_cam"]]
        if body_in_cam is not None:
            world_cam = [rotation[r] + [translation[r]] for r in range(3)]
            world_cam.append([0.0, 0.0, 0.0, 1.0])
            world_body = matmul4(world_cam, body_in_cam)
            rotation = [row[:3] for row in world_body[:3]]
            translation = [row[3] for row in world_body[:3]]
        tx, ty, tz = translation
        qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation)
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
    parser.add_argument("--wslam", type=Path, action="append", default=[],
                        help="wslam map.json (ExportMap sidecar, repeatable)")
    parser.add_argument("--cam-to-body", type=Path,
                        help="EuRoC cam0/sensor.yaml: convert wslam camera "
                             "poses to the body frame with its T_BS")
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
    parser.add_argument("--report", action="store_true",
                        help="also print the accuracy report "
                             "(accuracy_report.py: coverage, Umeyama scale, "
                             "APE under Sim(3)/SE(3)/4-DoF, RPE over 1/5/10 m, "
                             "local scale, gravity, time offset, loop pairs) "
                             "and write it to <out-dir>/accuracy_report.md")
    args = parser.parse_args()

    if not args.wslam and not args.orb and args.gt is None:
        parser.error("nothing to do: pass at least one of --wslam/--orb/--gt")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    estimates = []
    # (label, .tum, map json or None) per estimate, for the report
    report_inputs = []
    cam_in_body = (read_euroc_t_bs(args.cam_to_body)
                   if args.cam_to_body is not None else None)
    for wslam_path in args.wslam:
        stem = "wslam" if len(args.wslam) == 1 else wslam_path.stem
        out_path = args.out_dir / (stem + ".tum")
        convert_wslam_map_json(wslam_path, out_path, cam_in_body)
        estimates.append(out_path)
        report_inputs.append((stem, out_path, wslam_path))
    for orb_path in args.orb:
        out_path = args.out_dir / (orb_path.stem + ".tum")
        convert_orbslam_txt(orb_path, out_path)
        estimates.append(out_path)
        report_inputs.append((orb_path.stem, out_path, None))

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

    if args.report:
        write_accuracy_report(groundtruth_tum, report_inputs, cam_in_body,
                              args.out_dir / "accuracy_report.md")
    return 0 if all_ok else 1


def write_accuracy_report(groundtruth_tum: Path,
                          report_inputs: list[tuple[str, Path, Path | None]],
                          cam_in_body: list[list[float]] | None,
                          out_path: Path) -> None:
    """Print accuracy_report.py's report and write it to out_path."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import accuracy_report

    estimates = [accuracy_report.EstimateInputs(label, tum, map_json)
                 for label, tum, map_json in report_inputs]
    text = accuracy_report.build_report(
        groundtruth_tum, estimates,
        np.asarray(cam_in_body) if cam_in_body is not None else None)
    print("\n=== accuracy report ===\n")
    print(text)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote accuracy report -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
