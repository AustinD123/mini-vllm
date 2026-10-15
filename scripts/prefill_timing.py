"""Prefill step time and TTFT on the real model: the same script runs on two checkouts.

Each config admits n requests of prompt length L with a one-token output, so the first engine step is
the whole prefill and emits each request's first token. That step's wall time is the prefill step time,
and it is the TTFT of every request in it (all arrive at 0). Each repeat builds a fresh engine.

Run from the root of the checkout to time, so the code under test is the one imported:

    cd <checkout> && python <repo>/scripts/prefill_timing.py --label before --out <repo>/results/3050/prefill_before.json

The record includes the minivllm path it imported, the commit, and the first generated tokens, so two runs
can be compared for identical output.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.getcwd())
import minivllm  # noqa: E402
from minivllm.attention.backend import BACKEND_NAME, register_backend  # noqa: E402
from minivllm.bench.results import environment_info  # noqa: E402
from minivllm.engine import Engine  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIGS = [(1, 64), (1, 256), (8, 64), (8, 256)]  # (requests, prompt length)
REPEATS = 7
WARM = 2
BLOCK = 16


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def prefill_once(paged, cfg, n: int, length: int) -> tuple[float, dict[int, int]]:
    gen = torch.Generator().manual_seed(0)
    eng = Engine(
        paged, num_blocks=n * (-(-(length + 1) // BLOCK)) + 8, block_size=BLOCK, max_batch=n,
        token_budget=n * length, num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads, dtype=torch.float16,
    )
    for i in range(n):
        eng.add_request(i, torch.randint(1, cfg.vocab_size, (length,), generator=gen).tolist(), 1, arrival_time=0.0)
    torch.cuda.synchronize()
    t = time.perf_counter()
    eng.step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t) * 1e3
    return elapsed, {i: eng.tokens[i][length] for i in range(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    register_backend()
    paged = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, attn_implementation=BACKEND_NAME).eval().to("cuda")
    cfg = paged.config
    results = []
    for n, length in CONFIGS:
        for _ in range(WARM):
            prefill_once(paged, cfg, n, length)
        times, first = [], None
        for _ in range(REPEATS):
            ms, tokens = prefill_once(paged, cfg, n, length)
            times.append(ms)
            first = tokens
        results.append({"requests": n, "prompt_len": length, "median_ms": statistics.median(times),
                        "min_ms": min(times), "max_ms": max(times), "all_ms": times, "first_tokens": first})
        print(f"n={n} L={length}: median {results[-1]['median_ms']:.2f} ms [{min(times):.2f}-{max(times):.2f}]", flush=True)

    record = {
        "label": args.label,
        "minivllm_imported_from": str(Path(minivllm.__file__).resolve().parent),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty_tracked_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "environment": {**environment_info(), "cuda": torch.version.cuda},
        "settings": {"model": MODEL_ID, "dtype": "float16", "repeats": REPEATS, "warm": WARM, "block_size": BLOCK},
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
