"""Log a `MapSnapshot` to a rerun session.

The functions here are split into small helpers (`init`, `log_camera`,
`log_landmarks`) so the same pieces can be reused when incremental
streaming gets added: the caller would just wrap the per-keyframe log
calls in `rr.set_time_sequence("keyframe", kf.id)` and feed entities
as they appear, instead of looping over a fully loaded snapshot.
"""

from __future__ import annotations

import logging

import numpy as np
import rerun as rr

from .types import CameraIntrinsics, Keyframe, MapSnapshot

logger = logging.getLogger(__name__)

# Entity-path conventions chosen so the rerun viewer's tree groups the
# scene the way a SLAM operator expects: everything lives under /world,
# the trajectory and individual keyframe frustums hang off it, and the
# point cloud is a sibling.
_WORLD = "world"
_LANDMARKS = f"{_WORLD}/landmarks"
_KEYFRAMES = f"{_WORLD}/keyframes"
_TRAJECTORY = f"{_WORLD}/trajectory"


def init(application_id: str = "uni_slam", *, spawn: bool = True,
         coordinate_system: str = "RDF") -> None:
    """Start (or attach to) a rerun session and set the world axes."""
    rr.init(application_id, spawn=spawn)
    view_coords = _resolve_view_coordinates(coordinate_system)
    rr.log(_WORLD, view_coords, static=True)


def _resolve_view_coordinates(name: str) -> rr.ViewCoordinates:
    """Map our schema's `coordinate_system` string to a rerun archetype."""
    try:
        return getattr(rr.ViewCoordinates, name)
    except AttributeError:
        logger.warning(
            "unknown coordinate_system %r; falling back to RDF (OpenCV "
            "convention)",
            name,
        )
        return rr.ViewCoordinates.RDF


def log_landmarks(positions: np.ndarray, ids: np.ndarray, *,
                  entity: str = _LANDMARKS, static: bool = True) -> None:
    """Log the landmark point cloud. `static=True` means the cloud lives
    outside the timeline — set False when streaming per-frame updates.
    """
    rr.log(
        entity,
        rr.Points3D(positions=positions, class_ids=ids),
        static=static,
    )


def log_camera(kf: Keyframe, cam: CameraIntrinsics, *,
               entity_prefix: str = _KEYFRAMES) -> None:
    """Log a keyframe pose and its pinhole intrinsics as a frustum."""
    path = f"{entity_prefix}/{kf.id}"
    rr.log(
        path,
        rr.Transform3D(
            translation=kf.t_world_cam.astype(np.float32),
            mat3x3=kf.R_world_cam.astype(np.float32),
        ),
    )
    rr.log(
        f"{path}/image",
        rr.Pinhole(
            focal_length=[cam.fx, cam.fy],
            principal_point=[cam.cx, cam.cy],
            resolution=[cam.width, cam.height],
        ),
    )


def log_trajectory(keyframes: list[Keyframe], *,
                   entity: str = _TRAJECTORY) -> None:
    """Log the polyline through the keyframe positions in chronological order."""
    if len(keyframes) < 2:
        return
    points = np.stack(
        [kf.t_world_cam.astype(np.float32) for kf in keyframes], axis=0
    )
    rr.log(entity, rr.LineStrips3D(strips=[points]), static=True)


def show(snap: MapSnapshot, *, application_id: str = "uni_slam",
         spawn: bool = True) -> None:
    """Convenience: log everything in `snap` to a fresh rerun session.

    For the incremental-loading use case, call `init` once and then
    `log_landmarks` / `log_camera` / `log_trajectory` directly as data
    arrives, with `rr.set_time_sequence(...)` driving the timeline.
    """
    init(application_id, spawn=spawn,
         coordinate_system=snap.coordinate_system)

    log_landmarks(snap.landmark_positions, snap.landmark_ids)
    # Sort keyframes by id so the trajectory polyline is ordered even if
    # the input wasn't. iSAM2's calculateEstimate() returns Values in key
    # order, which already matches id order — this is a safety net.
    keyframes_sorted = sorted(snap.keyframes, key=lambda kf: kf.id)
    for kf in keyframes_sorted:
        log_camera(kf, snap.camera)
    log_trajectory(keyframes_sorted)

    logger.info(
        "logged map: %d landmarks, %d keyframes (factors=%s)",
        snap.landmark_positions.shape[0],
        len(snap.keyframes),
        snap.stats.get("factors", "?"),
    )
