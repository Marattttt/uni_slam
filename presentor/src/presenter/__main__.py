"""CLI entry point: `python -m presenter <map.ply>`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .loader import load_map
from .viewer import show


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="presenter",
        description="Visualise a pc_wslam factor-graph map in rerun.",
    )
    parser.add_argument(
        "map_path",
        type=Path,
        help="Path to the .ply map written by ExportMap (the .json sidecar "
             "is found alongside it automatically).",
    )
    parser.add_argument(
        "--application-id",
        default="uni_slam",
        help="rerun application id (default: %(default)s).",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn a rerun viewer; assume one is already listening "
             "or that the session is being recorded.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        snap = load_map(args.map_path)
    except (FileNotFoundError, ValueError) as e:
        logging.getLogger("presenter").error("failed to load map: %s", e)
        return 1

    show(snap, application_id=args.application_id, spawn=not args.no_spawn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
