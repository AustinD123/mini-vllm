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
