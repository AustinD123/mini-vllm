"""WallClock: engine timing comes only from real elapsed time, never from the cost model.

The engine charges the P11 cost model after each step. On WallClock that charge must be a no-op.
"""

from __future__ import annotations

import time

import minivllm.engine as engine_module
from engine_helpers import make_engine, make_models, prompts
from minivllm.clock import WallClock


def test_wall_clock_charge_adds_no_time():
    clock = WallClock()
    before = clock.now()
    clock.charge(1000.0)
    assert clock.now() - before < 1.0


def test_engine_on_wall_clock_ignores_the_cost_model(device, monkeypatch):
    # If the engine charged the cost model on WallClock, one step would add 1e6 seconds.
    monkeypatch.setattr(engine_module, "step_latency", lambda *args: 1e9)
    _, paged = make_models(device=device)
    clock = WallClock()
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=clock)
    engine.add_request(0, prompts([4])[0], 3)
    start = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - start

    (log,) = engine.request_logs()
    assert len(log.token_times) == 3
    assert max(log.token_times) < 60.0
    assert max(log.token_times) <= clock.now()
    assert log.finish_time - log.arrival_time <= elapsed + 1.0
