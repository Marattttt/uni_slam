"""Plain dataclasses shared by the loader and the viewer.

These mirror the schema documented in `wslam/export.cpp` (format_version 1):
camera + keyframes + landmark arrays + run stats. Keeping them as plain
dataclasses (rather than rerun archetypes) means the viewer module is the
only place that imports `rerun`, so the loader stays trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole + radial-tangential distortion intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    model: str = "pinhole"
    distortion_model: str = "radtan"


@dataclass(frozen=True)
class Keyframe:
    """One camera pose in the factor graph, expressed in world coordinates."""

    id: int
    R_world_cam: np.ndarray  # shape (3, 3), float64
    t_world_cam: np.ndarray  # shape (3,), float64
    # Source-frame timestamp in nanoseconds. 0 means the export predates
    # the timestamp field (format_version=1 maps written before that change).
    timestamp_ns: int = 0


@dataclass(frozen=True)
class MapSnapshot:
    """The final estimate of the SLAM map produced by pc_wslam.

    `landmark_positions` and `landmark_ids` are aligned arrays: row i in
    `landmark_positions` corresponds to landmark id `landmark_ids[i]`.
    """

    format_version: int
    coordinate_system: str
    camera: CameraIntrinsics
    keyframes: list[Keyframe]
    landmark_positions: np.ndarray  # shape (N, 3), float32
    landmark_ids: np.ndarray         # shape (N,), uint32
    stats: dict = field(default_factory=dict)
