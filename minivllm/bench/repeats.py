"""Repeat wall-clock runs and aggregate them: median and min-max spread per sweep point.

Plumbing, not a numbered problem.

Operations
    aggregate_repeats(runs) -> list[dict]
        runs[k] is the list of records from repeat k (one run_sweep call). Records are matched on
        (system, rate). Each output record carries the median of every numeric metric, a `spread`
        entry (the min and max over repeats) for each metric in SPREAD_METRICS, and `repeats`.
        Raises ValueError when a repeat does not cover every point.

Why: wall-clock timing is noisy on a laptop GPU and on a shared Colab machine. One run is not
enough to read a difference from. The median is the number to compare; the spread says how
much to trust it.
"""

from __future__ import annotations

import copy
import statistics

SPREAD_METRICS = ("ttft_50", "ttft_99", "tpot_50", "throughput")


def aggregate_repeats(runs: list[list[dict]]) -> list[dict]:
    if not runs:
        raise ValueError("need at least one repeat")
    groups: dict[tuple[str, float], list[dict]] = {}
    for run in runs:
        for record in run:
            groups.setdefault((record["system"], record["workload"]["rate"]), []).append(record)
    out = []
    for key in sorted(groups, key=lambda k: (k[1], k[0])):
        records = groups[key]
        if len(records) != len(runs):
            raise ValueError(f"point {key} appears in {len(records)} of {len(runs)} repeats")
        aggregated = copy.deepcopy(records[0])
        for name, value in records[0]["metrics"].items():
            values = [r["metrics"][name] for r in records]
            if all(v is None for v in values):
                aggregated["metrics"][name] = None
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                aggregated["metrics"][name] = statistics.median(values)
        aggregated["spread"] = {}
        for name in SPREAD_METRICS:
            values = [r["metrics"][name] for r in records if r["metrics"][name] is not None]
            if values:
                aggregated["spread"][name] = [min(values), max(values)]
        aggregated["repeats"] = len(records)
        out.append(aggregated)
    return out
