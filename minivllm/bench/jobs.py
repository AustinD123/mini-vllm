"""Concrete jobs for the benchmark: workload requests with real token ids, shared by every system.

Plumbing, not a numbered problem.

Operations
    Job(request_id, arrival_time, prompt_ids, output_len)
    build_jobs(requests, vocab_size, seed) -> list[Job]
        turns workload Requests into Jobs. Every request's first shared_prefix_len
        tokens are the same; the rest are drawn per request. Same seed, same tokens.
        Token ids are in [1, vocab_size), so 0 stays free for padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minivllm.bench.workload import Request


@dataclass(frozen=True)
class Job:
    request_id: int
    arrival_time: float
    prompt_ids: tuple[int, ...]
    output_len: int


def build_jobs(requests: list[Request], vocab_size: int, seed: int) -> list[Job]:
    """Return one Job per request. Raises ValueError when the requests disagree on shared_prefix_len."""
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be at least 2, got {vocab_size}")
    if not requests:
        return []
    shared_len = requests[0].shared_prefix_len
    if any(r.shared_prefix_len != shared_len for r in requests):
        raise ValueError("all requests must share one shared_prefix_len")
    shared = np.random.default_rng([seed, 3]).integers(1, vocab_size, size=shared_len).tolist()
    jobs = []
    for r in requests:
        tail_rng = np.random.default_rng([seed, 4, r.request_id])
        tail = tail_rng.integers(1, vocab_size, size=r.prompt_len - shared_len).tolist()
        jobs.append(Job(r.request_id, r.arrival_time, tuple(shared + tail), r.output_len))
    return jobs
