"""Static PNG plots of a sweep: throughput, TTFT p99, TPOT p99 and goodput against request rate.

Plumbing, not a numbered problem.

Operations
    plot_sweep(records, out_dir) -> list[Path]
        one PNG per metric in METRICS, one line per system, from result records (bench.results)
    plot_prefix_sweep(records, out_dir) -> list[Path]
        engine vs engine_prefix on the shared-prefix workload: throughput, TTFT p99, KV in use
    plot_preemptions(records, out_dir) -> list[Path]
        engine preemptions against KV pool size, on the cost-model clock (labelled simulated)
    sweep_table(records) -> list[dict]
        one row per (system, rate) with the headline metrics, for a table view next to the plots

Repeats
    A record that came from aggregate_repeats carries a `spread` entry (min and max over repeats).
    Lines then show the median (the metric value) with error bars from the min to the max.

Style
    Light surface only (notebook output). Colors are the first three slots of the validated
    categorical palette in the dataviz reference (light). Aqua sits below 3:1 contrast on the
    light surface, so the relief rule applies: identity comes from the legend, and sweep_table
    is the table view that ships with every plot. End-of-line labels were dropped because they
    collide where lines converge.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to PNG files; no GUI backend is needed
import matplotlib.pyplot as plt  # noqa: E402

SYSTEM_ORDER = ("sequential", "static", "engine")
SYSTEM_COLORS = {"sequential": "#2a78d6", "static": "#eb6834", "engine": "#1baf7a"}
PREFIX_SYSTEMS = ("engine", "engine_prefix")
PREFIX_COLORS = {"engine": "#2a78d6", "engine_prefix": "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e5e4e0"

METRICS = (
    ("throughput", "Output throughput (tokens/s)"),
    ("ttft_99", "TTFT p99 (s)"),
    ("tpot_99", "TPOT p99 (s)"),
    ("goodput", "Goodput (requests/s meeting both SLOs)"),
)
REPEAT_NOTE = "Median of wall-clock repeats; bars show the min to the max over repeats."


def _points(records: list[dict], system: str, metric: str) -> list[tuple[float, float, list[float] | None]]:
    """(rate, value, spread) per point, sorted by rate. Points with no value are dropped."""
    points = []
    for r in records:
        if r["system"] != system or r["metrics"].get(metric) is None:
            continue
        spread = r.get("spread", {}).get(metric)
        points.append((r["workload"]["rate"], r["metrics"][metric], spread))
    return sorted(points, key=lambda p: p[0])


def _draw_line(ax, records: list[dict], system: str, metric: str, color: str) -> bool:
    """Draw one system's line for one metric. Returns False when there is nothing to draw."""
    points = _points(records, system, metric)
    if not points:
        return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if all(p[2] is not None for p in points):
        lower = [y - p[2][0] for y, p in zip(ys, points)]
        upper = [p[2][1] - y for y, p in zip(ys, points)]
        ax.errorbar(xs, ys, yerr=[lower, upper], color=color, linewidth=2, marker="o", markersize=8,
                    capsize=4, elinewidth=1.2, label=system)
    else:
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=system)
    return True


def _style(ax, fig, xlabel: str, ylabel: str, title: str, note: str | None) -> None:
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="best", fontsize=9)
    if note:
        fig.text(0.01, 0.01, note, color=INK_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))


def plot_sweep(records: list[dict], out_dir: str | Path) -> list[Path]:
    """Write one PNG per metric into out_dir and return the paths. Creates out_dir when missing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = [s for s in SYSTEM_ORDER if any(r["system"] == s for r in records)]
    repeated = any("spread" in r for r in records)
    paths: list[Path] = []
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
        for system in systems:
            _draw_line(ax, records, system, metric, SYSTEM_COLORS[system])
        _style(ax, fig, "Request rate (requests/s)", label, label, REPEAT_NOTE if repeated else None)
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
        row = {
            "rate": r["workload"]["rate"],
            "system": r["system"],
            "throughput": m["throughput"],
            "ttft_p50": m["ttft_50"],
            "ttft_p99": m["ttft_99"],
            "tpot_p50": m["tpot_50"],
            "tpot_p99": m["tpot_99"],
            "goodput": m["goodput"],
            "output_tokens": m["output_tokens"],
            "repeats": r.get("repeats", 1),
        }
        spread = r.get("spread", {})
        for name in ("throughput", "ttft_50", "ttft_99", "tpot_50"):
            row[f"{name}_range"] = spread.get(name)
        rows.append(row)
    order = {name: i for i, name in enumerate(SYSTEM_ORDER + ("engine_prefix",))}
    return sorted(rows, key=lambda row: (row["rate"], order.get(row["system"], len(order))))


def plot_prefix_sweep(records: list[dict], out_dir: str | Path) -> list[Path]:
    """Engine vs engine with prefix caching, on a shared-prefix workload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repeated = any("spread" in r for r in records)
    panels = (
        ("throughput", "Output throughput (tokens/s)"),
        ("ttft_99", "TTFT p99 (s)"),
        ("kv_in_use_peak_fraction", "KV in use, peak (fraction of pool)"),
    )
    paths: list[Path] = []
    for metric, label in panels:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
        for system in PREFIX_SYSTEMS:
            _draw_line(ax, records, system, metric, PREFIX_COLORS[system])
        _style(ax, fig, "Request rate (requests/s)", label,
               f"Shared-prefix workload: {label}", REPEAT_NOTE if repeated else None)
        path = out_dir / f"prefix_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_preemptions(records: list[dict], out_dir: str | Path) -> list[Path]:
    """Engine preemptions against KV pool size (blocks). Simulated: the cost-model clock, one run per size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
    for system in PREFIX_SYSTEMS:
        points = sorted(
            (r["workload"]["num_blocks"], r["metrics"]["num_preemptions"])
            for r in records if r["system"] == system and r["metrics"]["num_preemptions"] is not None
        )
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=PREFIX_COLORS[system], linewidth=2, marker="o", markersize=8, label=system)
    _style(ax, fig, "KV pool size (blocks)", "Preemptions",
           "Preemptions against pool size (simulated)",
           "Simulated: cost-model clock, so the counts repeat exactly. Measures scheduler policy, not GPU speed.")
    path = out_dir / "preemptions_vs_pool.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]
