import copy

import pytest

from minivllm.bench.results import make_result, validate_result

METRICS = {f"{name}_{p}": 0.1 for name in ("ttft", "tpot", "itl", "e2e") for p in (50, 90, 99)}
METRICS.update(
    throughput=1.0,
    goodput=1.0,
    output_tokens=4,
    num_requests=2,
    cached_tokens=0,
    prefix_hit_rate=0.0,
    submitted=2,
    finished=2,
    rejected=0,
    oversized=0,
    num_preemptions=0,
    kv_in_use_peak_blocks=2,
    kv_in_use_peak_fraction=0.25,
    kv_occupancy_peak_blocks=3,
    kv_occupancy_peak_fraction=0.375,
)
ENVIRONMENT = {"gpu": None, "torch": "x", "transformers": "x", "python": "x", "git_commit": None}
BASELINE_METRICS = dict(METRICS, num_preemptions=None, kv_in_use_peak_blocks=None, kv_in_use_peak_fraction=None,
                        kv_occupancy_peak_blocks=None, kv_occupancy_peak_fraction=None)


def record(fraction: float = 0.5, system: str = "engine", metrics: dict | None = None) -> dict:
    rec = make_result(system, {"rate": 2.0, "num_blocks": 8}, 0, dict(metrics or METRICS), kv_memory_fraction=fraction)
    rec["environment"] = dict(ENVIRONMENT)
    return rec


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


# review-era rule: a record is only valid when every submitted request finished
def test_validate_rejects_unfinished_requests():
    bad = dict(METRICS, finished=1)
    with pytest.raises(ValueError, match="finished"):
        validate_result(record(metrics=bad))


def test_validate_rejects_negative_rejected_count():
    with pytest.raises(ValueError, match="rejected"):
        validate_result(record(metrics=dict(METRICS, rejected=-1)))


def test_engine_record_must_report_kv_numbers():
    with pytest.raises(ValueError, match="must report"):
        validate_result(record(metrics=dict(METRICS, kv_in_use_peak_blocks=None)))


def test_in_use_cannot_exceed_occupancy_or_the_pool():
    with pytest.raises(ValueError, match="in use"):
        validate_result(record(metrics=dict(METRICS, kv_in_use_peak_blocks=4)))
    with pytest.raises(ValueError, match="in use"):
        validate_result(record(metrics=dict(METRICS, kv_occupancy_peak_blocks=9)))


def test_baseline_record_has_no_kv_numbers():
    validate_result(record(system="sequential", metrics=BASELINE_METRICS))
    with pytest.raises(ValueError, match="None"):
        validate_result(record(system="sequential", metrics=METRICS))
