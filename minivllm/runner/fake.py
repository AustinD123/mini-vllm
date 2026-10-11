"""Fake runner cost model: how long one scheduler step takes, without a GPU.

Problem 11 — Cost model

Story
    The simulator needs a step time for every scheduler decision, but running a
    real model for every experiment is slow and needs a GPU. A cost model turns
    a step's work into a time estimate. Batching, chunking, and preemption
    decisions can then be compared in seconds on a laptop.

Operations
    step_latency(num_prefill_tokens, num_decode_seqs, total_context) -> float
        estimated step time in milliseconds

Model (fixed by the spec)
    latency = overhead + b * num_prefill_tokens + c * num_decode_seqs + d * total_context

    The coefficients below are starting guesses. Problem 26 fits them against
    measured steps, so keep them as module-level constants.

Errors
    Raises ValueError when any argument is negative.

Example (with the guesses below)
    step_latency(0, 0, 0)     -> 5.0           overhead only
    step_latency(100, 0, 0)   -> 5.0 + 0.05 * 100 = 10.0
    step_latency(0, 10, 0)    -> 5.0 + 0.2 * 10 = 7.0

Constraints
    Runs in O(1) time.
    The result must be strictly larger when any one argument increases,
    holding the others fixed (the coefficients are positive).

Edge cases to think about
    What is the step time with no work at all?
    Is the model linear, so that doubling the prefill tokens adds the same
    amount as the first doubling did?
    What should happen with a negative argument?

Tests
    tests/test_p11_cost_model.py covers monotonicity in each argument, zero work
    returning the overhead only, and rejecting negative arguments.
"""

from __future__ import annotations

OVERHEAD_MS = 5.0
PREFILL_MS_PER_TOKEN = 0.05
DECODE_MS_PER_SEQ = 0.2
CONTEXT_MS_PER_TOKEN = 0.0001


def step_latency(num_prefill_tokens: int, num_decode_seqs: int, total_context: int) -> float:
    """Return the estimated step time in milliseconds.

    Raises ValueError when any argument is negative.
    """
    if num_prefill_tokens < 0 or num_decode_seqs < 0 or total_context < 0:
        raise ValueError("All arguments must be non-negative")
    return OVERHEAD_MS + PREFILL_MS_PER_TOKEN * num_prefill_tokens + DECODE_MS_PER_SEQ * num_decode_seqs + CONTEXT_MS_PER_TOKEN * total_context
