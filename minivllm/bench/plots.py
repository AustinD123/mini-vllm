"""Static PNG plots of a sweep: throughput, TTFT p99, TPOT p99 and goodput against request rate.

Plumbing, not a numbered problem.

Operations
    plot_sweep(records, out_dir) -> list[Path]
        one PNG per metric in METRICS, one line per system, from result records (bench.results)
    sweep_table(records) -> list[dict]
        one row per (system, rate) with the headline metrics, for a table view next to the plots

Style
    Light surface only (notebook output). Colors are the first three slots of the validated
    categorical palette in the dataviz reference (light). Aqua sits below 3:1 contrast on the
    light surface, so the relief rule applies: identity comes from the legend, and sweep_table
    is the table view that ships with every plot. End-of-line labels were dropped because they
    collide where lines converge.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

SYSTEM_ORDER = ("sequential", "static", "engine")
SYSTEM_COLORS = {"sequential": "#2a78d6", "static": "#eb6834", "engine": "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e5e4e0"

METRICS = (
    ("throughput", "Output throughput (tokens/s)"),
    ("ttft_99", "TTFT p99 (s)"),
    ("tpot_99", "TPOT p99 (s)"),
    ("goodput", "Goodput (requests/s meeting both SLOs)"),
)


def _series(records: list[dict], system: str, metric: str) -> tuple[list[float], list[float]]:
    points = sorted(
        (r["workload"]["rate"], r["metrics"][metric]) for r in records if r["system"] == system
    )
    return [x for x, _ in points], [y for _, y in points]


def plot_sweep(records: list[dict], out_dir: str | Path) -> list[Path]:
    """Write one PNG per metric into out_dir and return the paths. Creates out_dir when missing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = [s for s in SYSTEM_ORDER if any(r["system"] == s for r in records)]
    paths: list[Path] = []
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
        ax.set_facecolor("white")
        for system in systems:
            xs, ys = _series(records, system, metric)
            ys = [math.nan if y is None else y for y in ys]
            color = SYSTEM_COLORS[system]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=system)
        ax.set_xlabel("Request rate (requests/s)", color=INK_SECONDARY)
        ax.set_ylabel(label, color=INK_SECONDARY)
        ax.set_title(label, color=INK_PRIMARY, loc="left", fontsize=11)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="best", fontsize=9)
        fig.tight_layout()
        path = out_dir / f"sweep_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def sweep_table(records: list[dict]) -> list[dict]:
    """Headline numbers per (system, rate), sorted by rate then system order."""
    rows = []
    for r in records:
        m = r["metrics"]
        rows.append(
            {
                "rate": r["workload"]["rate"],
                "system": r["system"],
                "throughput": m["throughput"],
                "ttft_p99": m["ttft_99"],
                "tpot_p99": m["tpot_99"],
                "goodput": m["goodput"],
                "output_tokens": m["output_tokens"],
            }
        )
    return sorted(rows, key=lambda row: (row["rate"], SYSTEM_ORDER.index(row["system"])))
