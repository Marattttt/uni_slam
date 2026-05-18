"""Read the PLY + JSON pair emitted by `wslam::ExportMap`.

Only ASCII PLY with the schema documented in `wslam/export.cpp` is handled —
that is, one vertex per landmark with `x`, `y`, `z` (float) and a
`landmark_id` (uint) property. plyfile is permissive about extra properties
so any forward-compatible addition stays loadable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from .types import CameraIntrinsics, Keyframe, MapSnapshot

_SUPPORTED_FORMAT_VERSION = 1


def _derive_metadata_path(ply_path: Path) -> Path:
    """Mirror the C++ side: replace `.ply` with `.json`."""
    if ply_path.suffix.lower() != ".ply":
        msg = f"map path must end in .ply, got {ply_path.suffix!r}"
        raise ValueError(msg)
    return ply_path.with_suffix(".json")


def _load_landmarks(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions Nx3 float32, ids N uint32) from the PLY file."""
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        msg = f"PLY at {ply_path} has no 'vertex' element"
        raise ValueError(msg)
    vertex = ply["vertex"]
    positions = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ]
    )
    if "landmark_id" not in vertex.data.dtype.names:
        msg = (
            f"PLY at {ply_path} is missing the 'landmark_id' property "
            "expected by ExportMap"
        )
        raise ValueError(msg)
    ids = np.asarray(vertex["landmark_id"], dtype=np.uint32)
    if positions.shape[0] != ids.shape[0]:
        msg = "landmark_id and position arrays disagree in length"
        raise ValueError(msg)
    return positions, ids


def _parse_camera(payload: dict) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        k1=float(payload.get("k1", 0.0)),
        k2=float(payload.get("k2", 0.0)),
        p1=float(payload.get("p1", 0.0)),
        p2=float(payload.get("p2", 0.0)),
        model=str(payload.get("model", "pinhole")),
        distortion_model=str(payload.get("distortion_model", "radtan")),
    )


def _parse_keyframe(payload: dict) -> Keyframe:
    rot = np.asarray(payload["R_world_cam"], dtype=np.float64)
    trans = np.asarray(payload["t_world_cam"], dtype=np.float64)
    if rot.shape != (3, 3):
        msg = f"R_world_cam for keyframe {payload.get('id')} must be 3x3, got {rot.shape}"
        raise ValueError(msg)
    if trans.shape != (3,):
        msg = f"t_world_cam for keyframe {payload.get('id')} must be (3,), got {trans.shape}"
        raise ValueError(msg)
    return Keyframe(id=int(payload["id"]), R_world_cam=rot, t_world_cam=trans)


def load_map(ply_path: str | Path) -> MapSnapshot:
    """Load both files of an exported map and return a `MapSnapshot`.

    The JSON sidecar path is derived from the PLY path. Use this when you
    want the full map at once; the function is intentionally split into
    small helpers so a future incremental-load entry point (streaming
    landmark batches as iSAM2 produces them) can reuse them.
    """
    ply_path = Path(ply_path)
    meta_path = _derive_metadata_path(ply_path)

    positions, ids = _load_landmarks(ply_path)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    version = int(meta.get("format_version", -1))
    if version != _SUPPORTED_FORMAT_VERSION:
        msg = (
            f"unsupported map format_version {version}; this presenter "
            f"speaks version {_SUPPORTED_FORMAT_VERSION}"
        )
        raise ValueError(msg)

    declared_ids = np.asarray(meta.get("landmark_ids", []), dtype=np.uint32)
    if declared_ids.shape != ids.shape or not np.array_equal(declared_ids, ids):
        msg = (
            f"landmark_ids in JSON do not match PLY ordering "
            f"({declared_ids.shape} vs {ids.shape}); refusing to load a "
            "potentially misaligned map"
        )
        raise ValueError(msg)

    return MapSnapshot(
        format_version=version,
        coordinate_system=str(meta.get("coordinate_system", "RDF")),
        camera=_parse_camera(meta["camera"]),
        keyframes=[_parse_keyframe(kf) for kf in meta.get("keyframes", [])],
        landmark_positions=positions,
        landmark_ids=ids,
        stats=dict(meta.get("stats", {})),
    )
