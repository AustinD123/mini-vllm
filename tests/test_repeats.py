import pytest

from minivllm.bench.repeats import SPREAD_METRICS, aggregate_repeats
from minivllm.bench.results import validate_result


def rec(system: str, rate: float, ttft_50: float, throughput: float, tpot_50=None) -> dict:
    metrics = {name: 0.0 for name in SPREAD_METRICS}
    metrics.update(ttft_50=ttft_50, throughput=throughput, tpot_50=tpot_50, ttft_99=ttft_50 * 2,
                   submitted=4, finished=4, num_preemptions=None, output_tokens=20)
    return {"system": system, "workload": {"rate": rate}, "metrics": metrics}


def test_median_and_min_max_spread_over_three_repeats():
    runs = [
        [rec("engine", 2.0, ttft_50=0.10, throughput=10.0, tpot_50=0.02)],
        [rec("engine", 2.0, ttft_50=0.30, throughput=14.0, tpot_50=0.05)],
        [rec("engine", 2.0, ttft_50=0.20, throughput=12.0, tpot_50=0.03)],
    ]
    (point,) = aggregate_repeats(runs)
    assert point["repeats"] == 3
    assert point["metrics"]["ttft_50"] == pytest.approx(0.20)  # median
    assert point["metrics"]["throughput"] == pytest.approx(12.0)
    assert point["spread"]["ttft_50"] == pytest.approx([0.10, 0.30])
    assert point["spread"]["throughput"] == pytest.approx([10.0, 14.0])
    assert point["metrics"]["num_preemptions"] is None  # all-None metrics stay None
    assert "num_preemptions" not in point["spread"]


def test_points_are_matched_by_system_and_rate():
    runs = [
        [rec("engine", 2.0, 0.1, 10.0), rec("static", 2.0, 0.5, 8.0)],
        [rec("engine", 2.0, 0.3, 12.0), rec("static", 2.0, 0.7, 6.0)],
    ]
    points = aggregate_repeats(runs)
    assert [(p["system"], p["workload"]["rate"]) for p in points] == [("engine", 2.0), ("static", 2.0)]
    assert points[1]["metrics"]["ttft_50"] == pytest.approx(0.6)


def test_a_repeat_missing_a_point_is_refused():
    runs = [
        [rec("engine", 2.0, 0.1, 10.0)],
        [rec("engine", 2.0, 0.1, 10.0), rec("static", 2.0, 0.5, 8.0)],
    ]
    with pytest.raises(ValueError, match="repeats"):
        aggregate_repeats(runs)


def test_plots_draw_median_with_spread_bars(tmp_path):
    from minivllm.bench.plots import plot_sweep

    runs = [[rec("engine", 2.0, 0.1, 10.0, 0.02), rec("sequential", 2.0, 0.5, 8.0, 0.04)],
            [rec("engine", 2.0, 0.3, 12.0, 0.05), rec("sequential", 2.0, 0.7, 6.0, 0.06)]]
    paths = plot_sweep(aggregate_repeats(runs), tmp_path)
    assert all(p.read_bytes()[:4] == bytes([0x89]) + b"PNG" for p in paths)


def test_validate_rejects_an_inverted_spread():
    from test_results_schema import record as valid_record

    rec_ = valid_record()
    rec_["spread"] = {"ttft_50": [0.5, 0.1]}
    with pytest.raises(ValueError, match="spread"):
        validate_result(rec_)
