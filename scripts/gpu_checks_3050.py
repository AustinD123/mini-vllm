"""The GPU checks from the Colab notebook, run on the local RTX 3050 (4 GB), each saved as JSON.

Writes results/3050/ with commit, GPU and versions in every file:
    agreement_fp16.json, agreement_fp32.json
        free-running agreement with HF generate, first divergence, teacher-forced agreement
    sizing_fp16.json
        KV pool blocks predicted from free memory, against bytes measured on the GPU
    identical_outputs_gate.json
        sequential, static and engine (batched and per-sequence) give the same greedy tokens

Limits on a 4 GB card, recorded in each file's "notes":
    - Only one model is resident at a time. The reference outputs are produced first, the
      reference model is freed, then the paged model is loaded to check them.
    - The identical-outputs gate runs the tiny float64 model, the same as the notebook's dry run.
      Two float64 copies of Qwen2.5-0.5B do not fit in 4 GB. The real-model gate stays on the T4.
    - The notebook's memory-pressure check (agreement_pressure.json) is not run here.

Run from the repo root:

    python scripts/gpu_checks_3050.py
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, ".")
from minivllm.attention.backend import BACKEND_NAME, register_backend  # noqa: E402
from minivllm.bench.gpu_checks import agreement, identical_outputs_gate, run_engine, sizing_check, teacher_forced_agreement  # noqa: E402
from minivllm.bench.jobs import build_jobs  # noqa: E402
from minivllm.bench.results import environment_info  # noqa: E402
from minivllm.bench.workload import LENGTH_DEFAULTS, make_workload  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
AGREEMENT_REQUESTS = 16
MAX_LEN = 256  # the notebook's CONFIG["max_len"]
MAX_BATCH = 16
TOKEN_BUDGET = 512
NUM_BLOCKS = 512
OUT_DIR = Path("results") / "3050"
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def env() -> dict:
    return {**environment_info(), "cuda": torch.version.cuda, "free_mib_at_start": round(torch.cuda.mem_get_info()[0] / 2**20)}


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def write(name: str, record: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps({**record, "env": ENV}, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT_DIR / name, flush=True)


def agreement_run(label: str, dtype: torch.dtype, jobs, prompts, lens) -> None:
    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa").eval().to("cuda")
    ref_out = {}
    for j in jobs:
        ids = torch.tensor([list(j.prompt_ids)], device="cuda")
        with torch.no_grad():
            out = ref.generate(ids, do_sample=False, max_new_tokens=j.output_len, min_new_tokens=j.output_len,
                               eos_token_id=None, pad_token_id=0, repetition_penalty=1.0)
        ref_out[j.request_id] = out[0, len(j.prompt_ids):].tolist()
    del ref
    release()

    register_backend()
    paged = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation=BACKEND_NAME).eval().to("cuda")
    engine_out, preemptions = run_engine(paged, prompts, lens, num_blocks=NUM_BLOCKS, max_batch=MAX_BATCH,
                                         token_budget=TOKEN_BUDGET)
    free_run = agreement(engine_out, ref_out)
    teacher = teacher_forced_agreement(paged, prompts, ref_out)
    write(f"agreement_{label}.json", {
        "model": MODEL_ID, "dtype": label, "max_len": MAX_LEN, "requests": len(jobs), "seed": SEED,
        "free_running": free_run, "teacher_forced": teacher, "preemptions": preemptions,
        "notes": "free_running feeds back each system's own tokens, so one divergence can cascade. "
                 "teacher_forced feeds the reference tokens and scores each position alone.",
    })
    del paged
    release()


def sizing_run() -> None:
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    free_before = round(torch.cuda.mem_get_info()[0] / 2**20)
    result = sizing_check(cfg, torch.float16, fraction=0.5)
    write("sizing_fp16.json", {
        "model": MODEL_ID, "dtype": "fp16", "free_mib_before": free_before, **result,
        "notes": "measured_peak_bytes is torch.cuda.max_memory_allocated around the KV pool allocation.",
    })


def gate_run() -> None:
    # The tiny float64 model from the notebook's dry run. Same seeds and workload generator.
    tiny = dict(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None)
    torch.manual_seed(SEED)
    ref = Qwen2ForCausalLM(Qwen2Config(**tiny, attn_implementation="sdpa")).eval().double().to("cuda")
    paged = Qwen2ForCausalLM(Qwen2Config(**tiny, attn_implementation=BACKEND_NAME)).eval().double().to("cuda")
    paged.load_state_dict(ref.state_dict())
    requests = make_workload(rate=10.0, num_requests=AGREEMENT_REQUESTS, seed=SEED, max_len=MAX_LEN, **LENGTH_DEFAULTS)
    jobs = build_jobs(requests, vocab_size=50, seed=SEED)
    gate = identical_outputs_gate(ref, paged, jobs, max_batch=MAX_BATCH, token_budget=TOKEN_BUDGET, num_blocks=NUM_BLOCKS)
    write("identical_outputs_gate.json", {
        "model": "tiny Qwen2, float64, vocab 50, 2 layers (not the real model)", "seed": SEED, **gate,
        "notes": "Same gate as the notebook dry run. The real model in float64 needs about 8 GB for two copies, "
                 "which the 3050 (4 GB) cannot hold.",
    })
    del ref, paged
    release()


def main() -> None:
    global ENV
    ENV = env()
    assert torch.cuda.is_available(), "this script needs the GPU"
    print("GPU:", ENV["gpu"], "| torch", ENV["torch"], "| commit", ENV["git_commit"], flush=True)

    requests = make_workload(rate=10.0, num_requests=AGREEMENT_REQUESTS, seed=SEED, max_len=MAX_LEN, **LENGTH_DEFAULTS)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    jobs = build_jobs(requests, vocab_size=cfg.vocab_size, seed=SEED)
    prompts = [list(j.prompt_ids) for j in jobs]
    lens = [j.output_len for j in jobs]

    agreement_run("fp16", torch.float16, jobs, prompts, lens)
    agreement_run("fp32", torch.float32, jobs, prompts, lens)
    sizing_run()
    gate_run()


if __name__ == "__main__":
    main()
