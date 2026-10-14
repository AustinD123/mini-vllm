"""Prefix diagnostics on the real model: prompt tokens forwarded, steps, TTFT, TPOT, wall time."""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from minivllm.attention.backend import BACKEND_NAME, register_backend  # noqa: E402
from minivllm.bench.jobs import build_jobs  # noqa: E402
from minivllm.bench.metrics import percentile, tpot, ttft  # noqa: E402
from minivllm.bench.workload import make_workload  # noqa: E402
from minivllm.clock import FakeClock, WallClock  # noqa: E402
from minivllm.engine import Engine  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
register_backend()
paged = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, attn_implementation=BACKEND_NAME).eval().to("cuda")
cfg = paged.config
requests = make_workload(rate=6.0, num_requests=24, seed=0, prompt_mu=3.0, prompt_sigma=0.4,
                         output_mu=3.0, output_sigma=0.4, shared_prefix_len=64, max_len=64)
jobs = build_jobs(requests, vocab_size=cfg.vocab_size, seed=0)
prompt_total = sum(len(j.prompt_ids) for j in jobs)


def run(prefix: bool, clock):
    eng = Engine(paged, num_blocks=512, block_size=16, max_batch=8, token_budget=512,
                 num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
                 head_dim=cfg.hidden_size // cfg.num_attention_heads, dtype=torch.float16,
                 clock=clock, prefix_cache=prefix)
    for j in jobs:
        eng.add_request(j.request_id, list(j.prompt_ids), j.output_len, arrival_time=j.arrival_time)
    counts = {"prompt_forwarded": 0, "forward_calls": 0}

    plain_step = eng.scheduler.step

    def counting_step():
        out = plain_step()  # admission (and cache credit) happens inside
        for sid, n in out.num_tokens.items():
            seq = eng.seqs[sid]
            before = seq.num_computed_tokens
            counts["prompt_forwarded"] += max(0, min(before + n, seq.prompt_len) - before)
        return out

    eng.scheduler.step = counting_step
    plain_forward = eng.runner.forward

    def counting_forward(*a, **k):
        counts["forward_calls"] += 1
        return plain_forward(*a, **k)

    eng.runner.forward = counting_forward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    while eng.has_work():
        eng.step()
        steps += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    logs = eng.request_logs()
    ttfts = [ttft(log) for log in logs]
    tpots = [tpot(log) for log in logs if len(log.token_times) >= 2]
    return {
        "cache": prefix,
        "clock": type(clock).__name__,
        "prompt_tokens_total": prompt_total,
        "cached_tokens": eng.cached_tokens,
        "prompt_forwarded": counts["prompt_forwarded"],
        "forward_calls": counts["forward_calls"],
        "steps": steps,
        "preemptions": eng.num_preemptions,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_p50": percentile(tpots, 50),
        "wall_s": wall,
        "outputs": {j.request_id: eng.tokens[j.request_id][len(j.prompt_ids):] for j in jobs},
    }


run(False, FakeClock())  # warm-up, discarded
results = []
for clock_factory in (FakeClock, WallClock):
    off = run(False, clock_factory())
    on = run(True, clock_factory())
    identical = off.pop("outputs") == on.pop("outputs")
    results.append({"off": off, "on": on, "identical_tokens": identical})
    print(json.dumps(results[-1], indent=1), flush=True)
with open("prefix_diag_results.json", "w") as f:
    json.dump(results, f, indent=1)
