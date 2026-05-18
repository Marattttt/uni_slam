# presenter

Small Python viewer for the factor-graph maps written by
`wslam::ExportMap`. Renders the landmark point cloud and the keyframe
trajectory in [rerun](https://www.rerun.io/), which is designed around
incremental `log(...)` calls — so streaming live updates from a running
pipeline is a small extension on top of this.

## What it loads

`presenter` consumes the two-file pair produced by `ExportMap`:

- `map.ply` — landmark point cloud (`x`, `y`, `z`, plus a `landmark_id`
  property)
- `map.json` — keyframes, camera intrinsics, run stats, schema
  (the sidecar path is derived by replacing the `.ply` extension)

The expected JSON schema is `format_version: 1`. See `wslam/export.cpp`
for the writer.

## Install

The project uses a `pyproject.toml` (PEP 621), so any modern installer
works. Pick one:

```bash
# uv (recommended — fast, supports the standard layout)
cd presentor
uv venv
uv pip install -e .

# or plain pip in a venv
cd presentor
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Python 3.10 or newer is required (matches `rerun-sdk`'s floor).

## Run

```bash
# from the repo root, with a map already exported (see pc_wslam --map-out=)
presenter /tmp/uniwslam_map.ply
```

The viewer launches as a separate process. Pass `--no-spawn` if you
want to attach to an already-running `rerun` viewer or record a session
file.

## Project layout

```
presentor/
  pyproject.toml
  README.md
  src/presenter/
    __init__.py     # re-exports
    __main__.py     # `python -m presenter` / `presenter` CLI
    types.py        # dataclasses: CameraIntrinsics, Keyframe, MapSnapshot
    loader.py       # PLY + JSON  → MapSnapshot
    viewer.py       # MapSnapshot → rerun log calls
```

`viewer.py` is split into `init`, `log_landmarks`, `log_camera`, and
`log_trajectory` helpers. The `show()` entry point composes them for
the final-map case; an incremental loader would call the same helpers
per arriving keyframe with `rr.set_time_sequence("keyframe", kf.id)`
driving the timeline.
