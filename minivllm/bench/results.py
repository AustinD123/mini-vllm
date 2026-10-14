"""Result records for benchmark runs: what ran, on what, with which seed and code.

Problem 25 — Benchmark sweep (plumbing)

Story
    A throughput number without its context is not credible. Every saved result
    records the GPU, the library versions, the git commit, and the seed, so a
    result can be traced back to the exact code and workload that produced it.

Operations
    environment_info() -> dict
        gpu, torch, transformers, python, and git_commit (None where unavailable)
    make_result(system, workload, seed, metrics) -> dict
        one record: system name, workload parameters, seed, metrics, and environment
    write_result(path, result) -> None
        writes the record as JSON
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def environment_info() -> dict:
    """Hardware and software versions for this run. gpu is None when CUDA is unavailable."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "gpu": gpu,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
    }


def make_result(system: str, workload: dict, seed: int, metrics: dict, kv_memory_fraction: float) -> dict:
    """One benchmark record. workload holds the parameters passed to make_workload.

    kv_memory_fraction is the share of free GPU memory the KV pool was sized from. Raises ValueError outside (0, 1].
    """
    _check_fraction(kv_memory_fraction)
    return {
        "system": system,
        "seed": seed,
        "workload": workload,
        "metrics": metrics,
        "kv_memory_fraction": kv_memory_fraction,
        "environment": environment_info(),
    }


def _check_fraction(value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ValueError(f"kv_memory_fraction must be a number in (0, 1], got {value!r}")


def write_result(path: str | Path, result: dict) -> None:
    """Write one record as indented JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


RESULT_KEYS = {"system", "seed", "workload", "metrics", "environment", "kv_memory_fraction"}
ENVIRONMENT_KEYS = {"gpu", "torch", "transformers", "python", "git_commit"}
METRIC_KEYS = {f"{name}_{p}" for name in ("ttft", "tpot", "itl", "e2e") for p in (50, 90, 99)} | {
    "throughput",
    "goodput",
    "output_tokens",
    "num_requests",
}


def validate_result(record: dict) -> None:
    """Raise ValueError when a result record is missing a required key. Extra keys are allowed."""
    missing = RESULT_KEYS - record.keys()
    if missing:
        raise ValueError(f"result is missing {sorted(missing)}")
    if not isinstance(record["system"], str) or not record["system"]:
        raise ValueError("system must be a non-empty string")
    if not isinstance(record["seed"], int):
        raise ValueError("seed must be an int")
    _check_fraction(record["kv_memory_fraction"])
    missing = METRIC_KEYS - record["metrics"].keys()
    if missing:
        raise ValueError(f"metrics is missing {sorted(missing)}")
    missing = ENVIRONMENT_KEYS - record["environment"].keys()
    if missing:
        raise ValueError(f"environment is missing {sorted(missing)}")
