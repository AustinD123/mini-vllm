"""Engine request events, fed through bench/metrics.py, checked against hand-computed fake-clock times.

Step cost (P11 cost model, ms) = 5 + 0.05 * prefill_tokens + 0.2 * decode_seqs + 0.0001 * context,
where context is the sum over scheduled sequences of (tokens computed before the step + tokens this step).
"""

from __future__ import annotations

import pytest

from engine_helpers import make_engine, make_models, prompts
from minivllm.bench.metrics import e2e, summarize, tpot, ttft
from minivllm.clock import FakeClock

# Prompt of 4 tokens, 3 output tokens, alone in the engine:
#   step 1: prefill 4, context 4          -> 5 + 0.2 + 0.0004 = 5.2004 ms
#   step 2: decode 1, context 4 + 1 = 5   -> 5 + 0.2 + 0.0005 = 5.2005 ms
#   step 3: decode 1, context 5 + 1 = 6   -> 5 + 0.2 + 0.0006 = 5.2006 ms
STEP_MS = [5.2004, 5.2005, 5.2006]


def test_ttft_and_tpot_match_hand_computed_fake_clock_times(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    engine.add_request(0, prompts([4])[0], 3)
    engine.run()

    (log,) = engine.request_logs()
    t1 = STEP_MS[0] / 1000
    t2 = (STEP_MS[0] + STEP_MS[1]) / 1000
    t3 = sum(STEP_MS) / 1000
    assert log.arrival_time == 0.0
    assert log.dequeue_time == 0.0
    assert log.token_times == pytest.approx([t1, t2, t3])
    assert log.finish_time == pytest.approx(t3)
    assert ttft(log) == pytest.approx(t1)
    assert tpot(log) == pytest.approx((t3 - t1) / 2)  # mean of the two gaps, 5.2005 ms each
    assert e2e(log) == pytest.approx(t3)


def test_ttft_counts_time_spent_queued_behind_another_request(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    first, second = prompts([4, 4])
    engine.add_request(0, first, 3)  # arrives at 0, finishes after its three steps
    engine.add_request(1, second, 1)  # arrives at 0, but max_batch=1 makes it wait for request 0
    engine.run()

    _, late = engine.request_logs()
    t_first_done = sum(STEP_MS) / 1000
    one_prefill = STEP_MS[0] / 1000
    assert late.arrival_time == 0.0  # scheduled arrival, not the moment it was dequeued
    assert late.dequeue_time == pytest.approx(t_first_done)
    assert ttft(late) == pytest.approx(t_first_done + one_prefill)
    assert e2e(late) == pytest.approx(t_first_done + one_prefill)


def test_summarize_uses_engine_logs(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    first, second = prompts([4, 4])
    engine.add_request(0, first, 3)
    engine.add_request(1, second, 1)
    engine.run()

    logs = engine.request_logs()
    summary = summarize(logs)
    duration = max(log.finish_time for log in logs) - min(log.arrival_time for log in logs)
    assert summary["throughput"] == pytest.approx(4 / duration)  # 3 + 1 output tokens
    # Nearest-rank p50 of two values is the smaller one.
    assert summary["ttft_50"] == pytest.approx(min(ttft(log) for log in logs))
