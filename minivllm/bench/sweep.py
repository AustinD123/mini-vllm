"""Sweep runner: every system at every request rate, on one seeded workload per rate.

Plumbing, not a numbered problem.

Operations
    run_sweep(ref_model, paged_model, *, rates, num_requests, seed, out_dir, ...) -> list[dict]
        ref_model    Hugging Face model with the default attention, used by the baselines
        paged_model  the same weights built with attn_implementation=BACKEND_NAME, used by the engine
        Runs every system in SYSTEMS at every rate and writes one JSON record per point to
        out_dir/<system>_rate<rate>.json. Returns the records in run order.

    run_point(system, ref_model, paged_model, jobs, clock, ...) -> (outputs, logs, cached_tokens)
        one system on one list of jobs. Each point gets a fresh clock from clock_factory.
        cached_tokens is the prompt tokens served from the prefix cache (0 unless engine_prefix).

Systems: sequential, static, engine (no prefix cache), engine_prefix (engine with PrefixCache).

Metrics per point come from bench.metrics.summarize, plus output_tokens, num_requests,
cached_tokens and prefix_hit_rate (cached prompt tokens over all prompt tokens).
Output tokens are the same for every system at one rate, because ignore-EOS makes each
request emit exactly output_len tokens.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from minivllm.attention.backend import register_backend
from minivllm.bench.baselines import run_sequential, run_static
from minivllm.bench.jobs import Job, build_jobs
from minivllm.bench.metrics import RequestLog, summarize
from minivllm.bench.results import make_result, write_result
from minivllm.bench.workload import LENGTH_DEFAULTS, make_workload
from minivllm.clock import Clock, FakeClock
from minivllm.engine import Engine
from minivllm.runner.sizing import num_blocks_for_memory

SYSTEMS = ("sequential", "static", "engine", "engine_prefix")
BLOCK_SIZE = 16


ENGINE_SYSTEMS = ("engine", "engine_prefix")


@dataclass
class PointResult:
    """What one system produced on one list of jobs. Engine-only fields are None for the baselines."""

    outputs: dict[int, list[int]]
    logs: list[RequestLog]
    submitted: int
    rejected: int
    cached_tokens: int = 0
    num_preemptions: int | None = None
    kv_in_use_peak: int | None = None
    kv_occupancy_peak: int | None = None


def run_point(
    system: str,
    ref_model,
    paged_model,
    jobs: list[Job],
    clock: Clock,
    *,
    max_batch: int,
    token_budget: int,
    num_blocks: int,
) -> PointResult:
    """Run one system over ``jobs``. Raises ValueError for an unknown system.

    The engine refuses a request it cannot hold; those count as rejected, not submitted.
    """
    if system == "sequential":
        outputs, logs = run_sequential(ref_model, jobs, clock)
        return PointResult(outputs, logs, submitted=len(jobs), rejected=0)
    if system == "static":
        outputs, logs = run_static(ref_model, jobs, clock, max_batch=max_batch)
        return PointResult(outputs, logs, submitted=len(jobs), rejected=0)
    if system in ENGINE_SYSTEMS:
        cfg = paged_model.config
        engine = Engine(
            paged_model,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            max_batch=max_batch,
            token_budget=token_budget,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            dtype=next(paged_model.parameters()).dtype,
            clock=clock,
            prefix_cache=system == "engine_prefix",
        )
        submitted = rejected = 0
        for job in jobs:
            try:
                engine.add_request(job.request_id, list(job.prompt_ids), job.output_len, arrival_time=job.arrival_time)
                submitted += 1
            except ValueError:
                rejected += 1
        outputs = engine.run()
        return PointResult(
            outputs,
            engine.request_logs(),
            submitted=submitted,
            rejected=rejected,
            cached_tokens=engine.cached_tokens,
            num_preemptions=engine.num_preemptions,
            kv_in_use_peak=engine.peak_in_use_blocks,
            kv_occupancy_peak=engine.peak_occupancy_blocks,
        )
    raise ValueError(f"unknown system {system!r}; expected one of {SYSTEMS}")


def _resolve_num_blocks(paged_model, num_blocks: int | None, kv_memory_fraction: float) -> int:
    if num_blocks is not None:
        return num_blocks
    if not torch.cuda.is_available():
        raise RuntimeError("num_blocks is None: sizing from kv_memory_fraction needs CUDA")
    free_bytes, _ = torch.cuda.mem_get_info()
    dtype = next(paged_model.parameters()).dtype
    return min(4096, num_blocks_for_memory(int(free_bytes * kv_memory_fraction), paged_model.config, dtype, BLOCK_SIZE))


def run_sweep(
    ref_model,
    paged_model,
    *,
    rates: list[float],
    num_requests: int,
    seed: int,
    out_dir: str | Path,
    systems: tuple[str, ...] = SYSTEMS,
    clock_factory: Callable[[], Clock] = FakeClock,
    vocab_size: int = 50,
    max_batch: int = 4,
    token_budget: int = 64,
    num_blocks: int | None = None,
    kv_memory_fraction: float = 0.5,
    length_overrides: dict | None = None,
) -> list[dict]:
    """Run every system at every rate. See the module docstring.

    length_overrides replaces LENGTH_DEFAULTS and any other make_workload arguments
    (for example max_len or shared_prefix_len). Use it to keep CPU runs small.

    KV pool size: when num_blocks is None, the pool is sized from kv_memory_fraction of the
    currently free GPU memory (CUDA only; RuntimeError otherwise). When num_blocks is given it is
    used as-is and kv_memory_fraction is recorded for provenance only. Every record carries
    kv_memory_fraction, and the workload carries num_blocks.

    Requests that cannot fit in the pool (prompt plus output longer than the pool's capacity) are
    dropped for every system, so the systems see the same requests. The count is recorded as
    oversized. Every point asserts that every submitted request finished.
    """
    register_backend()
    if not 0 < kv_memory_fraction <= 1:
        raise ValueError(f"kv_memory_fraction must be in (0, 1], got {kv_memory_fraction}")
    num_blocks = _resolve_num_blocks(paged_model, num_blocks, kv_memory_fraction)
    capacity = num_blocks * BLOCK_SIZE
    out_dir = Path(out_dir)
    lengths = {**LENGTH_DEFAULTS, **(length_overrides or {})}
    clock_name = type(clock_factory()).__name__  # FakeClock: cost-model time; WallClock: real time
    records: list[dict] = []
    for rate in rates:
        requests = make_workload(rate=rate, num_requests=num_requests, seed=seed, **lengths)
        all_jobs = build_jobs(requests, vocab_size=vocab_size, seed=seed)
        jobs = [j for j in all_jobs if len(j.prompt_ids) + j.output_len <= capacity]
        oversized = len(all_jobs) - len(jobs)
        workload = {"rate": rate, "num_requests": num_requests, "num_blocks": num_blocks,
                    "clock": clock_name, **lengths}
        for system in systems:
            point = run_point(
                system, ref_model, paged_model, jobs, clock_factory(),
                max_batch=max_batch, token_budget=token_budget, num_blocks=num_blocks,
            )
            finished = sum(1 for j in jobs if len(point.outputs.get(j.request_id, [])) == j.output_len)
            if finished != point.submitted:
                raise RuntimeError(f"{system} at rate {rate:g}: {finished} of {point.submitted} submitted requests finished")
            metrics = summarize(point.logs)
            prompt_tokens = sum(len(job.prompt_ids) for job in jobs)
            metrics["output_tokens"] = sum(len(tokens) for tokens in point.outputs.values())
            metrics["num_requests"] = point.submitted
            metrics["submitted"] = point.submitted
            metrics["finished"] = finished
            metrics["rejected"] = point.rejected
            metrics["oversized"] = oversized
            metrics["cached_tokens"] = point.cached_tokens
            metrics["prefix_hit_rate"] = point.cached_tokens / prompt_tokens if prompt_tokens else 0.0
            metrics["num_preemptions"] = point.num_preemptions
            engine_point = system in ENGINE_SYSTEMS
            metrics["kv_in_use_peak_blocks"] = point.kv_in_use_peak if engine_point else None
            metrics["kv_in_use_peak_fraction"] = point.kv_in_use_peak / num_blocks if engine_point else None
            metrics["kv_occupancy_peak_blocks"] = point.kv_occupancy_peak if engine_point else None
            metrics["kv_occupancy_peak_fraction"] = point.kv_occupancy_peak / num_blocks if engine_point else None
            record = make_result(system, workload, seed, metrics, kv_memory_fraction)
            write_result(out_dir / f"{system}_rate{rate:g}.json", record)
            records.append(record)
            del point
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def run_preemption_sweep(
    ref_model,
    paged_model,
    *,
    rate: float,
    num_requests: int,
    seed: int,
    out_dir: str | Path,
    pool_sizes: list[int],
    max_batch: int = 4,
    token_budget: int = 64,
    vocab_size: int = 50,
    clock_factory: Callable[[], Clock] = FakeClock,
    length_overrides: dict | None = None,
) -> list[dict]:
    """Engine preemptions against KV pool size, on one fixed workload.

    One record per pool size, with metrics.num_preemptions. Preemptions do not fall as the pool
    grows as quickly as one would expect: admission thrash, see the design log.
    """
    records: list[dict] = []
    for blocks in pool_sizes:
        records += run_sweep(
            ref_model, paged_model,
            rates=[rate], num_requests=num_requests, seed=seed, out_dir=Path(out_dir) / f"pool{blocks}",
            systems=("engine",), clock_factory=clock_factory, vocab_size=vocab_size,
            max_batch=max_batch, token_budget=token_budget, num_blocks=blocks,
            length_overrides=length_overrides,
        )
    return records
