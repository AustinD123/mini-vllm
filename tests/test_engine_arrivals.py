"""Timed arrivals: requests enter the scheduler at their arrival time, and an idle engine jumps forward.

The engine change itself (Engine._release_arrived) is human-written, so these fail until it lands.
"""

from __future__ import annotations

from engine_helpers import make_engine, make_models, prompts
from minivllm.bench.workload import make_workload
from minivllm.clock import FakeClock
from minivllm.core.sequence import SeqStatus


def test_no_request_is_scheduled_before_its_arrival(device):
    _, paged = make_models(device=device)
    clock = FakeClock()
    engine = make_engine(paged, num_blocks=8, max_batch=2, token_budget=64, clock=clock)
    early, late = prompts([3, 4])
    engine.add_request(0, early, 2, arrival_time=0.0)
    engine.add_request(1, late, 2, arrival_time=5.0)

    engine.step()  # t = 0: request 0 runs, request 1 has not arrived
    assert engine.seqs[1].status == SeqStatus.WAITING
    assert engine.seqs[1].num_computed_tokens == 0

    clock.advance_to(4.9)
    engine.step()
    assert engine.seqs[1].status == SeqStatus.WAITING
    assert engine.seqs[1].num_computed_tokens == 0

    clock.advance_to(5.0)
    engine.step()  # t = 5.0: request 1 is released and computed
    assert engine.seqs[1].num_computed_tokens > 0


def test_idle_engine_advances_to_next_arrival_without_spinning(device):
    _, paged = make_models(device=device)
    clock = FakeClock()
    engine = make_engine(paged, num_blocks=8, max_batch=2, token_budget=64, clock=clock)
    p0, p1 = prompts([3, 3])
    engine.add_request(0, p0, 1, arrival_time=5.0)  # one step: prefill plus its one output token
    engine.add_request(1, p1, 1, arrival_time=10.0)

    calls = 0
    plain_step = engine.step

    def counting_step() -> list[int]:
        nonlocal calls
        calls += 1
        return plain_step()

    engine.step = counting_step
    outputs = engine.run()

    assert set(outputs) == {0, 1}
    assert calls == 2, "an idle engine should jump straight to each arrival, not step through the gap"
    assert 10.0 <= clock.now() < 10.1


def test_same_seed_gives_identical_schedule(device):
    def schedule(seed: int) -> list:
        reqs = make_workload(
            rate=20.0, num_requests=6, seed=seed,
            prompt_mu=1.8, prompt_sigma=0.4, output_mu=1.0, output_sigma=0.3, max_len=12,
        )
        token_ids = prompts([r.prompt_len for r in reqs], seed=seed)
        _, paged = make_models(device=device)
        clock = FakeClock()
        engine = make_engine(paged, num_blocks=16, max_batch=3, token_budget=32, clock=clock)
        for r, p in zip(reqs, token_ids):
            engine.add_request(r.request_id, p, r.output_len, arrival_time=r.arrival_time)
        log: list = []
        plain_step = engine.step

        def recording_step() -> list[int]:
            finished = plain_step()
            running = tuple(sorted(seq.seq_id for seq in engine.scheduler.running))
            log.append((round(clock.now(), 9), running, tuple(sorted(finished))))
            return finished

        engine.step = recording_step
        engine.run()
        return log

    first = schedule(seed=7)
    assert first, "the schedule should not be empty"
    assert schedule(seed=7) == first
