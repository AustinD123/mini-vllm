"""Baselines: sequential and static HF batching against the engine on one seeded workload.

The gate: greedy decoding with EOS ignored, so sequential, static and engine must emit
byte-identical tokens. Outputs do not depend on timing, so the engine gets arrival 0 here;
the timed-arrival path is covered by test_engine_arrivals.py.
"""

from __future__ import annotations

from engine_helpers import make_engine, make_models
from minivllm.bench.baselines import run_sequential, run_static
from minivllm.bench.jobs import Job, build_jobs
from minivllm.bench.metrics import ttft
from minivllm.bench.workload import make_workload
from minivllm.clock import FakeClock


def seeded_jobs(seed: int = 4, num_requests: int = 6):
    requests = make_workload(
        rate=5.0, num_requests=num_requests, seed=seed,
        prompt_mu=2.0, prompt_sigma=0.5, output_mu=1.5, output_sigma=0.4,
        shared_prefix_len=8, max_len=20,
    )
    return build_jobs(requests, vocab_size=50, seed=seed)


def run_engine(paged, jobs, max_batch: int = 2):
    engine = make_engine(paged, num_blocks=16, max_batch=max_batch, token_budget=24, clock=FakeClock())
    for job in jobs:
        engine.add_request(job.request_id, list(job.prompt_ids), job.output_len)
    return engine.run()


def test_gate_sequential_static_and_engine_give_identical_outputs(device):
    ref, paged = make_models(device=device)
    jobs = seeded_jobs()
    sequential, _ = run_sequential(ref, jobs, FakeClock())
    static, _ = run_static(ref, jobs, FakeClock(), max_batch=3)
    engine = run_engine(paged, jobs)
    for job in jobs:
        assert len(sequential[job.request_id]) == job.output_len
    assert static == sequential
    assert engine == sequential


def test_single_request_gets_the_same_token_times_from_sequential_and_engine(device):
    ref, paged = make_models(device=device)
    job = Job(0, 0.0, (3, 17, 42, 8, 31), 4)
    _, (baseline_log,) = run_sequential(ref, [job], FakeClock())
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    engine.add_request(0, list(job.prompt_ids), job.output_len)
    engine.run()
    (engine_log,) = engine.request_logs()
    assert baseline_log.token_times == engine_log.token_times


def test_static_batch_starts_only_after_its_last_request_arrives(device):
    ref, _ = make_models(device=device)
    early = Job(0, 0.0, (3, 17, 42), 2)
    late = Job(1, 10.0, (5, 9), 2)
    _, logs = run_static(ref, [early, late], FakeClock(), max_batch=2)
    by_id = {log.request_id: log for log in logs}
    assert by_id[0].dequeue_time == 10.0  # the early request waits for the batch
    assert ttft(by_id[0]) > 10.0


# review: the Qwen instruct generation_config sets repetition_penalty=1.1, which generate() applies even under greedy
# decoding. The gate must still hold when the model carries that setting.
def test_gate_holds_when_the_model_carries_a_repetition_penalty(device):
    ref, paged = make_models(device=device)
    ref.generation_config.repetition_penalty = 1.1
    jobs = seeded_jobs()
    sequential, _ = run_sequential(ref, jobs, FakeClock())
    engine = run_engine(paged, jobs)
    assert engine == sequential
