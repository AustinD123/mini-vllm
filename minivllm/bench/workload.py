"""Seeded workload generator: Poisson arrivals, lognormal lengths, optional shared prefix.

Problem 25 — Benchmark sweep (plumbing)

Story
    A benchmark is only as good as its workload. Every system in the sweep has to
    see the same requests, arriving at the same times, with the same lengths, or the
    comparison means nothing. This module builds that workload from a seed, so any
    run can be reproduced exactly.

Operations
    poisson_arrivals(rate, num_requests, seed) -> np.ndarray
        cumulative arrival times in seconds, exponential gaps with mean 1 / rate
    lognormal_lengths(num_requests, mu, sigma, seed, min_len, max_len) -> np.ndarray
        integer token counts, clipped to [min_len, max_len]
    make_workload(rate, num_requests, seed, prompt_mu, prompt_sigma, output_mu,
                  output_sigma, shared_prefix_len=0, min_len=1, max_len=4096) -> list[Request]

Defaults
    LENGTH_DEFAULTS are placeholders, not fitted values. Replace them with stats
    fitted from the public trace you choose before reporting any number.

Note
    Each piece draws from its own generator derived from the seed, so changing one
    parameter does not reshuffle the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Log-scale mu and sigma fitted to ShareGPT_V3_unfiltered_cleaned_split.json (58,778 conversations),
# tokenized with Qwen/Qwen2.5-0.5B-Instruct, add_special_tokens=False.
# Prompt = first human message; output = first assistant reply. Untruncated fit.
LENGTH_DEFAULTS = {
    "prompt_mu": 3.835046,
    "prompt_sigma": 1.750131,
    "output_mu": 5.183535,
    "output_sigma": 1.160675,
}


@dataclass(frozen=True)
class Request:
    """One benchmark request. prompt_len includes any shared prefix."""

    request_id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    shared_prefix_len: int


def poisson_arrivals(rate: float, num_requests: int, seed: int) -> np.ndarray:
    """Cumulative arrival times for a Poisson process with the given rate (requests per second)."""
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if num_requests < 0:
        raise ValueError(f"num_requests must be non-negative, got {num_requests}")
    rng = np.random.default_rng([seed, 0])
    gaps = rng.exponential(1.0 / rate, size=num_requests)
    return np.cumsum(gaps)


def lognormal_lengths(
    num_requests: int,
    mu: float,
    sigma: float,
    seed: int,
    min_len: int = 1,
    max_len: int = 4096,
    stream: int = 1,
) -> np.ndarray:
    """Integer token counts drawn from a lognormal, clipped to [min_len, max_len]."""
    if min_len < 1 or max_len < min_len:
        raise ValueError(f"need 1 <= min_len <= max_len, got {min_len}, {max_len}")
    rng = np.random.default_rng([seed, stream])
    raw = rng.lognormal(mean=mu, sigma=sigma, size=num_requests)
    return np.clip(np.round(raw), min_len, max_len).astype(int)


def make_workload(
    rate: float,
    num_requests: int,
    seed: int,
    prompt_mu: float = LENGTH_DEFAULTS["prompt_mu"],
    prompt_sigma: float = LENGTH_DEFAULTS["prompt_sigma"],
    output_mu: float = LENGTH_DEFAULTS["output_mu"],
    output_sigma: float = LENGTH_DEFAULTS["output_sigma"],
    shared_prefix_len: int = 0,
    min_len: int = 1,
    max_len: int = 4096,
) -> list[Request]:
    """Build the full request list for one sweep point. Same seed, same list."""
    if shared_prefix_len < 0:
        raise ValueError(f"shared_prefix_len must be non-negative, got {shared_prefix_len}")
    arrivals = poisson_arrivals(rate, num_requests, seed)
    prompts = lognormal_lengths(num_requests, prompt_mu, prompt_sigma, seed, min_len, max_len, stream=1)
    outputs = lognormal_lengths(num_requests, output_mu, output_sigma, seed, min_len, max_len, stream=2)
    return [
        Request(
            request_id=i,
            arrival_time=float(arrivals[i]),
            prompt_len=int(prompts[i]) + shared_prefix_len,
            output_len=int(outputs[i]),
            shared_prefix_len=shared_prefix_len,
        )
        for i in range(num_requests)
    ]
