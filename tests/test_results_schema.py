import copy

import pytest

from minivllm.bench.results import make_result, validate_result

METRICS = {f"{name}_{p}": 0.1 for name in ("ttft", "tpot", "itl", "e2e") for p in (50, 90, 99)}
METRICS.update(throughput=1.0, goodput=1.0, output_tokens=4, num_requests=2, cached_tokens=0, prefix_hit_rate=0.0)


def record(fraction: float = 0.5) -> dict:
    return make_result("engine", {"rate": 2.0}, 0, dict(METRICS), kv_memory_fraction=fraction)


def test_make_result_records_the_kv_memory_fraction():
    rec = record(0.3)
    assert rec["kv_memory_fraction"] == 0.3
    validate_result(rec)


def test_validate_rejects_a_missing_fraction():
    rec = record()
    del rec["kv_memory_fraction"]
    with pytest.raises(ValueError, match="kv_memory_fraction"):
        validate_result(rec)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, "0.5", True, None])
def test_validate_rejects_an_out_of_range_or_non_numeric_fraction(bad):
    rec = copy.deepcopy(record())
    rec["kv_memory_fraction"] = bad
    with pytest.raises(ValueError):
        validate_result(rec)


def test_make_result_refuses_a_bad_fraction_up_front():
    with pytest.raises(ValueError):
        make_result("engine", {}, 0, dict(METRICS), kv_memory_fraction=0.0)
