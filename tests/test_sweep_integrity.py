"""Sweep integrity: oversized requests are dropped for every system and counted, every submitted
request finishes, and the preemption sweep records a count per pool size."""

from __future__ import annotations

from engine_helpers import make_models
from minivllm.bench.results import validate_result
from minivllm.bench.sweep import run_preemption_sweep, run_sweep

LENGTHS = dict(prompt_mu=3.0, prompt_sigma=0.4, output_mu=2.0, output_sigma=0.3, max_len=60)


def test_oversized_requests_are_dropped_for_every_system_and_counted(tmp_path):
    ref, paged = make_models()
    records = run_sweep(
        ref, paged, rates=[4.0], num_requests=12, seed=2, out_dir=tmp_path,
        systems=("sequential", "static", "engine"), max_batch=2, token_budget=32, num_blocks=3,
        length_overrides=LENGTHS,
    )
    assert len(records) == 3
    for record in records:
        validate_result(record)  # includes finished == submitted
    oversized = {r["metrics"]["oversized"] for r in records}
    assert len(oversized) == 1, "every system must see the same requests"
    assert oversized.pop() > 0, "the workload should include requests that cannot fit a 48-token pool"
    for record in records:
        assert record["metrics"]["finished"] == record["metrics"]["submitted"]
        assert record["metrics"]["rejected"] == 0


def test_preemption_sweep_records_a_count_per_pool_size(tmp_path):
    ref, paged = make_models()
    records = run_preemption_sweep(
        ref, paged, rate=20.0, num_requests=10, seed=3, out_dir=tmp_path, pool_sizes=[4, 16],
        max_batch=3, token_budget=32, length_overrides=dict(LENGTHS, max_len=40),
    )
    assert [r["workload"]["num_blocks"] for r in records] == [4, 16]
    for record in records:
        validate_result(record)
        assert isinstance(record["metrics"]["num_preemptions"], int)
