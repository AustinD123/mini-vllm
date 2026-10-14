from minivllm.bench.plots import METRICS, plot_sweep, sweep_table


def fake_record(system: str, rate: float) -> dict:
    metrics = {f"{name}_{p}": 0.1 for name in ("ttft", "tpot", "itl", "e2e") for p in (50, 90, 99)}
    metrics.update(throughput=10.0 * rate, goodput=rate / 2, output_tokens=40, num_requests=10)
    return {"system": system, "seed": 0, "workload": {"rate": rate}, "metrics": metrics, "environment": {}}


def test_plot_sweep_writes_one_png_per_metric(tmp_path):
    records = [fake_record(s, r) for s in ("sequential", "static", "engine") for r in (2.0, 8.0)]
    paths = plot_sweep(records, tmp_path)
    assert len(paths) == len(METRICS)
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_sweep_table_is_sorted_by_rate_then_system():
    records = [fake_record(s, r) for r in (8.0, 2.0) for s in ("engine", "sequential")]
    rows = sweep_table(records)
    assert [(row["rate"], row["system"]) for row in rows] == [
        (2.0, "sequential"), (2.0, "engine"), (8.0, "sequential"), (8.0, "engine"),
    ]
