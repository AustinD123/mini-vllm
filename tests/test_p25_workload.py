import json

import numpy as np
import pytest

from minivllm.bench.results import environment_info, make_result, write_result
from minivllm.bench.workload import make_workload, poisson_arrivals, lognormal_lengths


def test_same_seed_gives_same_workload():
    a = make_workload(rate=2.0, num_requests=50, seed=7)
    b = make_workload(rate=2.0, num_requests=50, seed=7)
    assert a == b


def test_different_seed_gives_different_arrivals():
    a = poisson_arrivals(rate=2.0, num_requests=50, seed=1)
    b = poisson_arrivals(rate=2.0, num_requests=50, seed=2)
    assert not np.array_equal(a, b)


def test_arrivals_are_increasing_and_mean_gap_matches_rate():
    arrivals = poisson_arrivals(rate=4.0, num_requests=5000, seed=3)
    assert np.all(np.diff(arrivals) > 0)
    mean_gap = np.mean(np.diff(np.concatenate([[0.0], arrivals])))
    assert mean_gap == pytest.approx(1 / 4.0, rel=0.05)


def test_changing_output_lengths_does_not_move_arrivals():
    a = make_workload(rate=2.0, num_requests=30, seed=5, output_mu=3.0)
    b = make_workload(rate=2.0, num_requests=30, seed=5, output_mu=6.0)
    assert [r.arrival_time for r in a] == [r.arrival_time for r in b]
    assert [r.output_len for r in a] != [r.output_len for r in b]


def test_lengths_are_positive_ints_within_bounds():
    lengths = lognormal_lengths(1000, mu=4.0, sigma=1.5, seed=9, min_len=2, max_len=300)
    assert lengths.dtype.kind == "i"
    assert lengths.min() >= 2
    assert lengths.max() <= 300


def test_shared_prefix_is_added_to_every_prompt():
    plain = make_workload(rate=1.0, num_requests=20, seed=4, shared_prefix_len=0)
    shared = make_workload(rate=1.0, num_requests=20, seed=4, shared_prefix_len=128)
    for p, s in zip(plain, shared):
        assert s.prompt_len == p.prompt_len + 128
        assert s.shared_prefix_len == 128


# extra edge case
def test_zero_rate_raises():
    with pytest.raises(ValueError):
        poisson_arrivals(rate=0.0, num_requests=10, seed=0)


# extra edge case
def test_min_len_above_max_len_raises():
    with pytest.raises(ValueError):
        lognormal_lengths(10, mu=1.0, sigma=1.0, seed=0, min_len=5, max_len=4)


def test_result_record_has_required_fields(tmp_path):
    env = environment_info()
    assert {"gpu", "torch", "transformers", "python", "git_commit"} <= set(env)
    result = make_result("mini-vllm", {"rate": 2.0}, seed=7, metrics={"tok_per_s": 123.4}, kv_memory_fraction=0.5)
    assert result["seed"] == 7
    assert result["system"] == "mini-vllm"
    path = tmp_path / "run.json"
    write_result(path, result)
    assert json.loads(path.read_text(encoding="utf-8")) == result
