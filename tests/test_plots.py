from minivllm.bench.plots import METRICS, plot_sweep, sweep_table


def fake_record(system: str, rate: float) -> dict:
    metrics = {f"{name}_{p}": 0.1 for name in ("ttft", "tpot", "itl", "e2e") for p in (50, 90, 99)}
    metrics.update(throughput=10.0 * rate, goodput=rate / 2, output_tokens=40, num_requests=10,
                   kv_in_use_peak_fraction=0.5, kv_occupancy_peak_fraction=0.75, num_preemptions=2,
                   kv_in_use_peak_blocks=4, kv_occupancy_peak_blocks=6)
    return {"system": system, "seed": 0, "workload": {"rate": rate, "num_blocks": 8}, "metrics": metrics, "environment": {}}


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


def test_prefix_sweep_plot_and_table_accept_engine_prefix(tmp_path):
    from minivllm.bench.plots import plot_prefix_sweep

    records = [fake_record(s, r) for s in ("engine", "engine_prefix") for r in (2.0, 8.0)]
    paths = plot_prefix_sweep(records, tmp_path)
    png_magic = b"\x89PNG\r\n\x1a\n"
    assert len(paths) == 3 and all(p.read_bytes()[:8] == png_magic for p in paths)
    rows = sweep_table(records)
    assert [row["system"] for row in rows[:2]] == ["engine", "engine_prefix"]


def test_preemption_plot_writes_a_png(tmp_path):
    from minivllm.bench.plots import plot_preemptions

    records = [fake_record("engine", 2.0)]
    (path,) = plot_preemptions(records, tmp_path)
    png_magic = bytes([0x89]) + b"PNG"
    assert path.read_bytes()[:4] == png_magic
