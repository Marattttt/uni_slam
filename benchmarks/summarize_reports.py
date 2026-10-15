"""Side-by-side table of the key metrics of several accuracy reports.

Each argument is LABEL=REPORT[#SECTION]: REPORT an accuracy_report.md that
compare.py --report wrote, SECTION the estimate's heading in it (its file
stem: map, map_vio, map_pgo, f_dataset-V103_monoi_r1, ...); without one the
first section is used. Prints a markdown table, one column per argument.

    python benchmarks/summarize_reports.py \\
        "VIO before=results/V103/before/eval/accuracy_report.md#map_vio" \\
        "VIO after=results/V103/after/eval/accuracy_report.md#map_vio"
"""

import re
import sys
from pathlib import Path

# (row label, metric line prefix, value extractor)
ROWS = [
    ("coverage", "coverage of the ground truth", r"^([\d.]+%)"),
    ("Umeyama scale", "**Umeyama scale**", r"\(([+-][\d.]+%)\)"),
    ("APE Sim(3) RMSE", "APE translation, Sim(3)", r"^([\d.]+ m)"),
    ("APE SE(3) RMSE", "APE translation, SE(3)", r"^([\d.]+ m)"),
    ("APE 4-DoF RMSE", "APE translation, position + yaw", r"^([\d.]+ m)"),
    ("APE Sim(3) max", "APE translation, Sim(3)", r"max ([\d.]+)"),
    ("gravity error (world)", "**gravity direction error, world frame**",
     r"^([\d.]+ deg)"),
    ("rotation APE (offset removed)", "APE rotation, SE(3), offset removed",
     r"^([\d.]+ deg)"),
    ("RPE 1 m (median)", "RPE over 1 m", r"\(([\d.]+%) median\) metric"),
    ("RPE 5 m (median)", "RPE over 5 m", r"\(([\d.]+%) median\) metric"),
    ("RPE 10 m (median)", "RPE over 10 m", r"\(([\d.]+%) median\) metric"),
    ("RPE 1 m rotation", "RPE over 1 m", r"rotation ([\d.]+ deg) RMSE"),
    ("local scale spread", "local scale", r"spread ([\d.]+%)"),
]


def read_section(path: Path, section: str | None) -> dict[str, str]:
    metrics: dict[str, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            if current is not None and metrics:
                break
            current = line[4:].strip()
            if section is not None and current != section:
                current = None
            continue
        if current is None or not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] not in ("metric", "---"):
            metrics[cells[0]] = cells[1]
    return metrics


def value(metrics: dict[str, str], prefix: str, pattern: str) -> str:
    for name, text in metrics.items():
        if name.startswith(prefix):
            match = re.search(pattern, text)
            return match.group(1) if match else "–"
    return "–"


def main() -> None:
    columns = []
    for argument in sys.argv[1:]:
        label, _, target = argument.partition("=")
        path, _, section = target.partition("#")
        report = Path(path)
        metrics = read_section(report, section or None) if report.exists() else {}
        columns.append((label, metrics))

    print("| metric | " + " | ".join(label for label, _ in columns) + " |")
    print("|---|" + "---|" * len(columns))
    for row, prefix, pattern in ROWS:
        cells = [value(metrics, prefix, pattern) if metrics else "failed"
                 for _, metrics in columns]
        print(f"| {row} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
