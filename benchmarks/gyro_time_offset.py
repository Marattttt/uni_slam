#!/usr/bin/env python3
"""Clock offset of a trajectory against the IMU, from rotation alone.

For each trajectory, the rotation angle over every k-pose window is compared
with the gyro integrated over the same window shifted by a lag; the lag with
the smallest RMS difference is the trajectory clock's offset from the IMU
clock. Rotation is scale-free, so a monocular (vision-only) trajectory works,
and nothing about the body/camera extrinsics is needed (angles only).

  lag > 0: the trajectory's pose at t matches the gyro at t + lag, i.e. its
           timestamps run behind the IMU clock (add lag to put them on it)

On UZH-FPV indoor_45_2 (2026-10-16): a vision-only ORB-SLAM3 run on camera
timestamps moved by Kalibr's timeshift_cam_imu shows 0 ms, so that shift is
right; groundtruth.txt shows -16 ms, so the ground truth runs ~16 ms ahead
of the IMU clock (about the raw camera clock), which compare.py's
--gt-time-offset corrects.

Usage: gyro_time_offset.py IMU_TXT LABEL=TRAJECTORY [LABEL=TRAJECTORY ...]
  IMU_TXT     — UZH-FPV imu.txt (id t[s] wx wy wz ax ay az)
  TRAJECTORY  — TUM-ordered `t tx ty tz qx qy qz qw`; timestamps in seconds,
                or nanoseconds (told apart by size, > 1e11)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation

NANOSECOND_THRESHOLD = 1e11
WINDOW_POSES = 3
LAGS_S = np.arange(-0.060, 0.0601, 0.001)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imu", help="UZH-FPV imu.txt")
    parser.add_argument("trajectories", nargs="+", metavar="LABEL=TRAJECTORY")
    args = parser.parse_args()

    imu = np.loadtxt(args.imu, comments="#")
    imu_stamps, gyro = imu[:, 1], imu[:, 2:5]
    # Trapezoidal running integral of the gyro, so any window is a difference
    integral = np.vstack([np.zeros(3), np.cumsum(
        0.5 * (gyro[1:] + gyro[:-1]) * np.diff(imu_stamps)[:, None], axis=0)])

    def gyro_angle(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        at = lambda t: np.stack([np.interp(t, imu_stamps, integral[:, i]) for i in range(3)], 1)
        return np.linalg.norm(at(end) - at(start), axis=1)

    for item in args.trajectories:
        label, path = item.split("=", 1)
        data = np.loadtxt(path, comments="#")
        stamps = data[:, 0]
        if stamps[0] > NANOSECOND_THRESHOLD:
            stamps = stamps / 1e9
        rotations = Rotation.from_quat(data[:, 4:8])
        angle = np.linalg.norm(
            (rotations[:-WINDOW_POSES].inv() * rotations[WINDOW_POSES:]).as_rotvec(), axis=1)
        start, end = stamps[:-WINDOW_POSES], stamps[WINDOW_POSES:]
        # Skip windows across tracking gaps; the small-angle sum of the gyro
        # matches the rotation angle only for short windows
        usable = (end > start) & (end - start < 0.2) & (angle < 0.5)
        errors = np.array([
            np.sqrt(np.mean((gyro_angle(start[usable] + lag, end[usable] + lag)
                             - angle[usable]) ** 2))
            for lag in LAGS_S])
        best = int(np.argmin(errors))
        zero = int(np.argmin(np.abs(LAGS_S)))
        print(f"{label}: lag {LAGS_S[best] * 1000:+.0f} ms "
              f"(RMS {errors[best] * 1e3:.2f} mrad, {errors[zero] * 1e3:.2f} at 0; "
              f"{usable.sum()} windows of {WINDOW_POSES} poses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
