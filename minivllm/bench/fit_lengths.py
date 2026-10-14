"""Fit lognormal length parameters from measured token counts.

Problem 25 — Benchmark sweep (plumbing)

Story
    The workload draws prompt and output lengths from lognormal distributions. Their
    parameters should come from real data, so the benchmark reflects real requests.
    This module turns a list of measured token counts into the mu and sigma that
    make_workload expects. Tokenizing the dataset is done separately, in the notebook,
    because it needs the model's tokenizer.

Operations
    fit_lognormal(lengths) -> (mu, sigma)
        lengths: token counts, all >= 1
        mu, sigma: the log-scale mean and standard deviation of the lengths

Errors
    Raises ValueError when lengths is empty or contains a value below 1.
"""

from __future__ import annotations

import numpy as np


def fit_lognormal(lengths) -> tuple[float, float]:
    """Return (mu, sigma) of the log of the lengths. Raises ValueError on empty or non-positive input."""
    values = np.asarray(lengths, dtype=float)
    if values.size == 0:
        raise ValueError("lengths must not be empty")
    if np.any(values < 1):
        raise ValueError("every length must be at least 1 token")
    logs = np.log(values)
    return float(logs.mean()), float(logs.std(ddof=0))
