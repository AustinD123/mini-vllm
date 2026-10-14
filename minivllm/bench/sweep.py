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
) -> tuple[dict[int, list[int]], list[RequestLog], int]:
    """Run one system over ``jobs``. Raises ValueError for an unknown system."""
    if system == "sequential":
        outputs, logs = run_sequential(ref_model, jobs, clock)
        return outputs, logs, 0
    if system == "static":
        outputs, logs = run_static(ref_model, jobs, clock, max_batch=max_batch)
        return outputs, logs, 0
    if system in ("engine", "engine_prefix"):
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
        for job in jobs:
            engine.add_request(job.request_id, list(job.prompt_ids), job.output_len, arrival_time=job.arrival_time)
        outputs = engine.run()
        return outputs, engine.request_logs(), engine.cached_tokens
    raise ValueError(f"unknown system {system!r}; expected one of {SYSTEMS}")


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
    """
    register_backend()
    if not 0 < kv_memory_fraction <= 1:
        raise ValueError(f"kv_memory_fraction must be in (0, 1], got {kv_memory_fraction}")
    if num_blocks is None:
        if not torch.cuda.is_available():
            raise RuntimeError("num_blocks is None: sizing from kv_memory_fraction needs CUDA")
        free_bytes, _ = torch.cuda.mem_get_info()
        dtype = next(paged_model.parameters()).dtype
        budget = int(free_bytes * kv_memory_fraction)
        num_blocks = min(4096, num_blocks_for_memory(budget, paged_model.config, dtype, BLOCK_SIZE))
    out_dir = Path(out_dir)
    lengths = {**LENGTH_DEFAULTS, **(length_overrides or {})}
    records: list[dict] = []
    for rate in rates:
        requests = make_workload(rate=rate, num_requests=num_requests, seed=seed, **lengths)
        jobs = build_jobs(requests, vocab_size=vocab_size, seed=seed)
        workload = {"rate": rate, "num_requests": num_requests, "num_blocks": num_blocks, **lengths}
        for system in systems:
            outputs, logs, cached = run_point(
                system, ref_model, paged_model, jobs, clock_factory(),
                max_batch=max_batch, token_budget=token_budget, num_blocks=num_blocks,
            )
            metrics = summarize(logs)
            prompt_tokens = sum(len(job.prompt_ids) for job in jobs)
            metrics["output_tokens"] = sum(len(tokens) for tokens in outputs.values())
            metrics["num_requests"] = len(outputs)
            metrics["cached_tokens"] = cached
            metrics["prefix_hit_rate"] = cached / prompt_tokens if prompt_tokens else 0.0
            record = make_result(system, workload, seed, metrics, kv_memory_fraction)
            write_result(out_dir / f"{system}_rate{rate:g}.json", record)
            records.append(record)
            del outputs, logs  # release per-point state before the next engine is built
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records
