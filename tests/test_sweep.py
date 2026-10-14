"""Sweep smoke test: 2 rates x 3 systems x 10 requests on the tiny CPU model.

The engine point needs timed arrivals (Engine._release_arrived), so these tests stay red
until step 2 is implemented.
"""

from __future__ import annotations

import json

from engine_helpers import make_models
from minivllm.bench.results import validate_result
from minivllm.bench.sweep import SYSTEMS, run_sweep

SMOKE = dict(
    rates=[2.0, 8.0],
    num_requests=10,
    seed=3,
    max_batch=3,
    token_budget=32,
    num_blocks=32,
    length_overrides=dict(
        prompt_mu=2.0, prompt_sigma=0.4, output_mu=1.3, output_sigma=0.3, shared_prefix_len=4, max_len=16
    ),
)


def test_smoke_sweep_writes_valid_records(tmp_path):
    ref, paged = make_models()
    records = run_sweep(ref, paged, out_dir=tmp_path, **SMOKE)

    assert len(records) == len(SMOKE["rates"]) * len(SYSTEMS)
    for record in records:
        validate_result(record)
    assert {r["system"] for r in records} == set(SYSTEMS)
    assert len(list(tmp_path.glob("*.json"))) == len(records)
    for path in tmp_path.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))

    # The same requests run through every system, so token counts agree at each rate.
    for rate in SMOKE["rates"]:
        at_rate = [r for r in records if r["workload"]["rate"] == rate]
        assert len({r["metrics"]["output_tokens"] for r in at_rate}) == 1
        assert all(r["metrics"]["num_requests"] == 10 for r in at_rate)


def test_rerun_with_same_seed_gives_identical_token_counts(tmp_path):
    ref, paged = make_models()
    first = run_sweep(ref, paged, out_dir=tmp_path / "a", **SMOKE)
    second = run_sweep(ref, paged, out_dir=tmp_path / "b", **SMOKE)

    def counts(records):
        return [(r["system"], r["workload"]["rate"], r["metrics"]["output_tokens"]) for r in records]

    assert counts(first) == counts(second)


SHARED_PREFIX = dict(
    rates=[4.0],
    num_requests=12,
    seed=5,
    max_batch=4,
    token_budget=64,
    num_blocks=32,
    length_overrides=dict(
        prompt_mu=2.0, prompt_sigma=0.3, output_mu=1.3, output_sigma=0.3, shared_prefix_len=32, max_len=16
    ),
)


def test_shared_prefix_sweep_point_reports_a_prefix_hit_rate(tmp_path):
    # Hashes register only after compute, so sharing needs requests that arrive over time.
    # The workload uses Poisson arrivals (never all at t=0), which is what makes caching measurable.
    from minivllm.bench.jobs import build_jobs
    from minivllm.bench.workload import make_workload

    requests = make_workload(rate=4.0, num_requests=12, seed=5, **SHARED_PREFIX["length_overrides"])
    arrivals = [r.arrival_time for r in requests]
    assert arrivals[0] > 0.0 and arrivals == sorted(arrivals) and len(set(arrivals)) == len(arrivals)
    assert all(j.prompt_ids[:32] == build_jobs(requests, 50, 5)[0].prompt_ids[:32] for j in build_jobs(requests, 50, 5))

    ref, paged = make_models()
    (record,) = run_sweep(ref, paged, out_dir=tmp_path, systems=("engine_prefix",), **SHARED_PREFIX)
    assert record["metrics"]["cached_tokens"] > 0
    assert record["metrics"]["prefix_hit_rate"] > 0
