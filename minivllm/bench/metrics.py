"""Latency and throughput metrics computed from a per-request event log.

Problem 12 — Metrics from an event log

Story
    A benchmark run produces, for every request, when it arrived, when the
    scheduler first picked it up, when each output token came out, and when it
    finished. Those raw times have to become the numbers people compare:
    time to first token, time between tokens, and tail percentiles. Users
    feel the tail, so p99 matters more than the average.

Units
    All times are in seconds. The cost model (problem 11) returns milliseconds,
    so convert before it reaches these functions.

Operations
    RequestLog(request_id, arrival_time, dequeue_time, token_times, finish_time)
        one request's events. arrival_time is when the request was scheduled to
        arrive, not when the scheduler dequeued it. token_times holds every
        output token's emit time, in order.
    ttft(log) -> float          time from scheduled arrival to the first token
    tpot(log) -> float          mean time between output tokens after the first
    itl(log) -> list[float]     each gap between consecutive output tokens
    e2e(log) -> float           time from scheduled arrival to finish
    percentile(values, p) -> float    nearest-rank percentile, p in 0..100
    summarize(logs, slo_ttft=2.0, slo_tpot=0.1) -> dict[str, float]
        keys: ttft_p50/p90/p99, tpot_p50/p90/p99, itl_p50/p90/p99,
              e2e_p50/p90/p99, throughput, goodput
        throughput: output tokens per second over the whole run
        goodput: requests per second that meet both SLOs (TTFT <= slo_ttft,
                 TPOT <= slo_tpot)

Choices (the spec leaves these open)
    Run duration = last finish time minus earliest scheduled arrival.
    Nearest-rank percentile: sort ascending, take the value at rank
    ceil(p / 100 * N), counting from 1.
    A request with one output token has no TPOT and no ITL; it still counts
    toward TTFT and E2E.

Errors
    tpot(log) raises ValueError when the request has fewer than 2 output tokens.
    percentile raises ValueError on an empty list.

Example
    log A: arrival 0.0, dequeue 1.0, tokens [2.0, 3.0, 5.0], finish 5.0
        ttft(A) -> 2.0       (measured from arrival, not from dequeue)
        tpot(A) -> 1.5       (5.0 - 2.0) / 2
        itl(A)  -> [1.0, 2.0]
        e2e(A)  -> 5.0

    percentile([1, 2, ..., 100], 50) -> 50
    percentile([1, 2, ..., 100], 99) -> 99

Constraints
    ttft, tpot, itl and e2e run in O(number of tokens).
    percentile runs in O(N log N) time.

Edge cases to think about
    Why must TTFT use the scheduled arrival rather than the dequeue time?
    What is TPOT for a request with exactly 2 tokens?
    What rank does the 90th percentile of 10 values fall on?
    Does a request that misses the TTFT SLO count toward goodput?

Tests
    tests/test_p12_metrics.py covers a hand-built log with known answers,
    TTFT measured from scheduled arrival, nearest-rank percentiles, the
    single-token TPOT error, and summarize's throughput and goodput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RequestLog:
    """One request's events. See the module docstring for the meaning of each field."""

    request_id: int
    arrival_time: float
    dequeue_time: float
    token_times: list[float]
    finish_time: float



def ttft(log: RequestLog) -> float:
    """Time from scheduled arrival to the first output token, in seconds."""
    if not log.token_times:
        raise ValueError("Request has no output tokens")
    return log.token_times[0] - log.arrival_time    


def tpot(log: RequestLog) -> float:
    """Mean time between output tokens after the first, in seconds.

    Raises ValueError when the request has fewer than 2 output tokens.

    """
    if len(log.token_times) < 2:
        raise ValueError("Request has fewer than 2 output tokens")
    gaps = [log.token_times[i] - log.token_times[i - 1] for i in range(1, len(log.token_times))]
    return sum(gaps) / len(gaps)    
    raise NotImplementedError


def itl(log: RequestLog) -> list[float]:
    """Each gap between consecutive output tokens, in seconds."""
    if len(log.token_times) < 2:
        raise ValueError("Request has fewer than 2 output tokens")
    return [log.token_times[i] - log.token_times[i - 1] for i in range(1, len(log.token_times))]
    raise NotImplementedError


def e2e(log: RequestLog) -> float:
    """Time from scheduled arrival to finish, in seconds."""
    return log.finish_time - log.arrival_time
    raise NotImplementedError


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile of ``values``.

    Raises ValueError when ``values`` is empty.
    """
    if not values:
        raise ValueError("Values cannot be empty")
    sorted_values = sorted(values)
    rank = math.ceil(p * len(sorted_values) / 100.0)
    index = max(rank - 1, 0)
    return sorted_values[index]


def summarize(
    logs: list[RequestLog], slo_ttft: float = 2.0, slo_tpot: float = 0.1
) -> dict[str, float]:
    """Return the summary keys listed in the module docstring.

    A percentile with no data behind it (for example TPOT when every request
    has one token) is reported as NaN rather than raising.
    """
    def safe_percentile(values: list[float], p: float) -> float:
        return percentile(values, p) if values else math.nan

    ttft_values = [ttft(log) for log in logs]
    tpot_values = [tpot(log) for log in logs if len(log.token_times) >= 2]
    itl_values = [gap for log in logs if len(log.token_times) >= 2 for gap in itl(log)]
    e2e_values = [e2e(log) for log in logs]
    duration = max(log.finish_time for log in logs) - min(log.arrival_time for log in logs)
    throughput = sum(len(log.token_times) for log in logs) / duration
    goodput = sum(
        1 for log in logs if ttft(log) <= slo_ttft and (len(log.token_times) < 2 or tpot(log) <= slo_tpot)
    ) / duration
    summary: dict[str, float] = {}
    for name, values in (("ttft", ttft_values), ("tpot", tpot_values), ("itl", itl_values), ("e2e", e2e_values)):
        for p in (50, 90, 99):
            summary[f"{name}_{p}"] = safe_percentile(values, p)
    summary["throughput"] = throughput
    summary["goodput"] = goodput
    return summary
