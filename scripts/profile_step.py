"""Profile one steady-state decode step of the real engine, and one HF forward on the same batch.

Diagnostic only: no engine code is changed. Run from the repo root:

    python scripts/profile_step.py --out profile_out

Writes profile_out/results.json and profile_out/decode_trace.json (Chrome trace).

Measurements (each on a fresh engine warmed to steady decode: every running sequence has
exactly one uncomputed token):
    wall      median step time, synced at the end of each step
    phases    CPU time per phase (no extra syncs): scheduler.step, runner prep, backend
              (all 24 layers), model non-attention, scheduler.update, sampling residual
    cprofile  cProfile over 20 steps, averaged per step, for the named functions
    trace     torch.profiler over 20 steps: top ops by CPU and CUDA time, kernel count,
              GPU idle inside the step annotations
    syncs     GPU-to-CPU sync points per step, from torch.cuda sync-debug warnings,
              grouped by the Python file:line that triggered them
    scaling   batch, context length and pool size, each run as a separate engine
    hf        HF forward at the same batch size, with a contiguous DynamicCache

Caveat: profiling adds overhead. Treat the phase split as attribution, not an exact budget.
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import collections
import json
import os
import pstats
import statistics
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AttentionInterface, AutoModelForCausalLM

sys.path.insert(0, ".")
from minivllm.attention import backend as backend_mod  # noqa: E402
from minivllm.attention.backend import BACKEND_NAME, SeqSlice, StepContext, paged_backend_forward, register_backend  # noqa: E402
from minivllm.attention.kv_pool import KVPool  # noqa: E402
from minivllm.core import invariants  # noqa: E402
from minivllm.core import prefix_cache as prefix_mod  # noqa: E402
from minivllm.engine import Engine  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 16
PERF = time.perf_counter


def sync() -> None:
    torch.cuda.synchronize()


def median(xs: list[float]) -> float:
    return statistics.median(xs)


BATCHED_DECODE = True  # set from --per-sequence: False runs the old per-sequence attention path


def steady_engine(model, cfg, *, num_blocks: int, batch: int, prompt_len: int, out_len: int,
                  prefix: bool = False) -> Engine:
    """Engine with `batch` requests, stepped until all of them are in steady decode."""
    eng = Engine(
        model, num_blocks=num_blocks, block_size=BLOCK, max_batch=batch, token_budget=max(512, 2 * batch),
        num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads, dtype=torch.float16, prefix_cache=prefix,
        batched_decode=BATCHED_DECODE,
    )
    rng = np.random.default_rng(0)
    for i in range(batch):
        eng.add_request(i, rng.integers(1, cfg.vocab_size, size=prompt_len).tolist(), out_len, arrival_time=0.0)
    for _ in range(2000):
        run = eng.scheduler.running
        if len(run) == batch and all(s.num_uncomputed() == 1 for s in run):
            break
        eng.step()
    else:
        raise RuntimeError("engine never reached steady decode")
    sync()
    return eng


def wall_times(eng: Engine, n: int = 20, warm: int = 3) -> list[float]:
    for _ in range(warm):
        eng.step()
    sync()
    out = []
    for _ in range(n):
        t = PERF()
        eng.step()
        sync()
        out.append((PERF() - t) * 1e3)
    return out


def phase_times(eng: Engine, n: int = 20) -> dict[str, float]:
    """Per-step CPU ms for each phase. Wrappers are removed before returning."""
    acc = collections.defaultdict(float)
    restore = []

    def wrap(obj, attr, label, instance=False):
        orig = getattr(obj, attr)

        def w(*a, **k):
            t = PERF()
            r = orig(*a, **k)
            acc[label] += PERF() - t
            return r

        setattr(obj, attr, w)
        restore.append((obj, attr, instance))

    wrap(eng, "step", "engine_step", instance=True)
    wrap(eng.scheduler, "step", "scheduler_step", instance=True)
    wrap(eng.scheduler, "update", "scheduler_update", instance=True)
    wrap(eng.runner, "forward", "runner_forward", instance=True)
    wrap(eng.runner.model, "forward", "model_forward", instance=True)

    orig_backend = paged_backend_forward

    def timed_backend(*a, **k):
        t = PERF()
        r = orig_backend(*a, **k)
        acc["backend_all_layers"] += PERF() - t
        acc["backend_calls"] += 1
        return r

    AttentionInterface.register(BACKEND_NAME, timed_backend)
    try:
        for _ in range(3):
            eng.step()
        sync()
        acc.clear()
        for _ in range(n):
            eng.step()
        sync()
    finally:
        AttentionInterface.register(BACKEND_NAME, orig_backend)
        for obj, attr, instance in restore:
            if instance:
                delattr(obj, attr)  # drop the instance-level wrapper, keep the class method
    out = {k: v * 1e3 / n for k, v in acc.items() if k != "backend_calls"}
    out["backend_calls_per_step"] = acc["backend_calls"] / n
    out["sampling_and_bookkeeping"] = out["engine_step"] - out["scheduler_step"] - out["runner_forward"] - out["scheduler_update"]
    out["runner_prep"] = out["runner_forward"] - out["model_forward"]
    out["model_non_attention"] = out["model_forward"] - out["backend_all_layers"]
    return out


@contextlib.contextmanager
def swapped(obj, name: str, fn):
    """Temporarily replace obj.name with fn. Works for module attributes and instance attributes."""
    orig = getattr(obj, name)
    setattr(obj, name, fn)
    try:
        yield
    finally:
        setattr(obj, name, orig)


def ablate_write_checks(eng: Engine):
    """Measurement only: KVPool.write without its two torch.any() range checks (48 syncs per step)."""
    kv = eng.kv

    def write_unchecked(layer, slots, k, v):
        bi = slots // kv.k.shape[2]
        oi = slots % kv.k.shape[2]
        kv.k[layer, bi, oi] = k
        kv.v[layer, bi, oi] = v

    return swapped(kv, "write", write_unchecked)


class _DeviceTensorMemo:
    """Stands in for the torch module inside the backend: torch.tensor(data, ..., device) is served
    from a cache, so the host-to-device copy happens once per distinct slot list, not once per layer.
    Everything else is forwarded to torch. Measurement only."""

    def __init__(self) -> None:
        self.cache: dict = {}

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, data, dtype=None, device=None):
        key = (tuple(data), dtype, str(device))
        hit = self.cache.get(key)
        if hit is None:
            hit = self.cache[key] = torch.tensor(data, dtype=dtype, device=device)
        return hit


def ablate_slot_device_memo(eng: Engine):
    """Measurement only: slot lists and their device copies reused across the 24 layers of one forward.
    Removes the per-layer host-to-device copy (and its sync), the largest sync source in the trace."""
    memo_obj = _DeviceTensorMemo()
    slot_ctx = ablate_slot_memo(eng)
    orig_forward = eng.runner.forward

    def forward_with_clear(*a, **k):
        memo_obj.cache.clear()
        return orig_forward(*a, **k)

    @contextlib.contextmanager
    def both():
        with slot_ctx, swapped(backend_mod, "torch", memo_obj), swapped(eng.runner, "forward", forward_with_clear):
            yield

    return both()


def ablate_slot_memo(eng: Engine):
    """Measurement only: slot_mapping's host list reused across the 24 layers of one forward.
    The per-layer host-to-device copy is still made, so this removes Python work, not syncs."""
    real = backend_mod.slot_mapping
    cache: dict = {}

    def memo(block_table, start, end, block_size):
        key = (tuple(block_table), start, end, block_size)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = real(block_table, start, end, block_size)
        return hit

    orig_forward = eng.runner.forward

    def forward_with_clear(*a, **k):
        cache.clear()
        return orig_forward(*a, **k)

    @contextlib.contextmanager
    def both():
        with swapped(backend_mod, "slot_mapping", memo), swapped(eng.runner, "forward", forward_with_clear):
            yield

    return both()


def ablations(paged, cfg, batch: int, prompt_len: int, out_len: int, n: int = 40, rounds: int = 3) -> dict:
    """Median step time with and without each ablation, interleaved so drift affects all arms equally."""
    # Each arm is a factory, because a context manager can only be entered once.
    eng = steady_engine(paged, cfg, num_blocks=4096, batch=batch, prompt_len=prompt_len, out_len=out_len)
    arms = {
        "baseline": lambda: contextlib.nullcontext(),
        "no_pool_write_checks": lambda: ablate_write_checks(eng),
        "slot_list_memo": lambda: ablate_slot_memo(eng),
        "slot_device_memo": lambda: ablate_slot_device_memo(eng),
    }
    samples = {k: [] for k in arms}
    for _ in range(rounds):
        for label, make in arms.items():
            with make():
                samples[label].append(median(wall_times(eng, n=n, warm=2)))
    out = {k: median(v) for k, v in samples.items()}
    out["samples_ms"] = samples
    return out


def invariant_and_hash_cost(eng: Engine, n: int = 20) -> dict[str, float]:
    """Cost of the checks that are off in the bench path, measured on the same live state."""
    t = PERF()
    for _ in range(n):
        invariants.check(eng.scheduler)
    inv = (PERF() - t) * 1e3 / n
    prompts = [eng.tokens[s.seq_id][:s.prompt_len] for s in eng.scheduler.running]
    t = PERF()
    for _ in range(n):
        for p in prompts:
            prefix_mod.block_hashes(p, BLOCK)
    hashes = (PERF() - t) * 1e3 / n
    return {"invariant_check_per_step_ms": inv, "rehash_all_running_prompts_per_step_ms": hashes}


def cprofile_step(eng: Engine, n: int = 20) -> dict[str, dict[str, float]]:
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        eng.step()
    sync()
    pr.disable()
    stats = pstats.Stats(pr).stats

    def pick(func: str, file_end: str | None = None) -> dict[str, float]:
        calls = tot = cum = 0.0
        for (fname, _, name), (_, nc, tt, ct, _) in stats.items():
            if name == func and (file_end is None or fname.replace("\\", "/").endswith(file_end)):
                calls += nc
                tot += tt
                cum += ct
        return {"calls_per_step": calls / n, "self_ms_per_step": tot * 1e3 / n, "cum_ms_per_step": cum * 1e3 / n}

    return {
        "Engine.step": pick("step", "minivllm/engine.py"),
        "Scheduler.step": pick("step", "minivllm/core/scheduler.py"),
        "Scheduler.update": pick("update", "minivllm/core/scheduler.py"),
        "invariants.check": pick("check", "minivllm/core/invariants.py"),
        "prefix block_hashes": pick("block_hashes", "minivllm/core/prefix_cache.py"),
        "RealRunner.forward (prep+model)": pick("forward", "minivllm/runner/real.py"),
        "block_table.slot_mapping": pick("slot_mapping", "minivllm/core/block_table.py"),
        "block_table.slot_for_position": pick("slot_for_position", "minivllm/core/block_table.py"),
        "paged_backend_forward": pick("paged_backend_forward", "minivllm/attention/backend.py"),
        "grouped_query_attention": pick("grouped_query_attention", "minivllm/attention/gqa.py"),
        "KVPool.write": pick("write", "minivllm/attention/kv_pool.py"),
    }


def trace_step(eng: Engine, out_dir: str, n: int = 20) -> dict:
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(3):
        eng.step()
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            with record_function("decode_step"):
                eng.step()
        sync()
    path = os.path.join(out_dir, "decode_trace.json")
    prof.export_chrome_trace(path)

    rows = []
    for item in prof.key_averages():
        cpu = getattr(item, "self_cpu_time_total", 0.0)
        dev = getattr(item, "self_device_time_total", None)
        if dev is None:
            dev = getattr(item, "self_cuda_time_total", 0.0)
        rows.append({"op": item.key, "count_per_step": item.count / n, "self_cpu_us_per_step": cpu / n, "self_device_us_per_step": dev / n})
    top_cpu = sorted(rows, key=lambda r: -r["self_cpu_us_per_step"])[:20]
    top_dev = sorted(rows, key=lambda r: -r["self_device_us_per_step"])[:20]

    with open(path) as f:
        events = json.load(f)["traceEvents"]
    steps = [(e["ts"], e["ts"] + e["dur"]) for e in events if e.get("ph") == "X" and e.get("name") == "decode_step"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    gpu_ops = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kernels_per_step = len(kernels) / n
    idle = busy = window = 0.0
    for s, e in steps:
        window += e - s
        ivs = sorted((max(g["ts"], s), min(g["ts"] + g["dur"], e)) for g in gpu_ops if g["ts"] < e and g["ts"] + g["dur"] > s)
        covered = 0.0
        cur_s = cur_e = None
        for a, b in ivs:
            if b <= a:
                continue
            if cur_e is None or a > cur_e:
                if cur_e is not None:
                    covered += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        if cur_e is not None:
            covered += cur_e - cur_s
        busy += covered
    idle = window - busy
    memcpy = [e for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy"]
    return {
        "top_cpu": top_cpu,
        "top_device": top_dev,
        "kernels_per_step": kernels_per_step,
        "memcpy_per_step": len(memcpy) / n,
        "step_window_ms_per_step": window / n / 1e3,
        "gpu_busy_ms_per_step": busy / n / 1e3,
        "gpu_idle_ms_per_step": idle / n / 1e3,
        "gpu_idle_fraction": idle / window if window else float("nan"),
    }


def sync_sites(eng: Engine, n: int = 20) -> dict:
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in range(n):
                eng.step()
            sync()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    sites = collections.Counter()
    for w in rec:
        if "synchronizing" in str(w.message):
            sites[f"{os.path.relpath(w.filename, os.getcwd())}:{w.lineno}".replace("\\", "/")] += 1
    per_step = {k: v / n for k, v in sites.most_common()}
    return {"syncs_per_step": sum(per_step.values()), "by_site_per_step": per_step}


def isolated_backend(num_layers: int, batch: int, out_dir: str, ctx: int = 100) -> dict:
    """Kernel count and CPU time of one paged_backend_forward call (one layer), at a given batch size."""
    from torch.profiler import ProfilerActivity, profile

    kv_heads, q_heads, head_dim = 2, 14, 64
    blocks_per_seq = -(-ctx // BLOCK)
    pool = KVPool(num_layers, batch * blocks_per_seq + 1, BLOCK, kv_heads, head_dim, dtype=torch.float16, device="cuda")
    tables = [list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq)) for i in range(batch)]
    last = ctx - 1
    slots = torch.tensor([t[last // BLOCK] * BLOCK + last % BLOCK for t in tables], dtype=torch.long, device="cuda")
    seqs = [SeqSlice(i, 1, tables[i], ctx) for i in range(batch)]
    ctx_obj = StepContext(pool=pool, slots=slots, seqs=seqs)
    q = torch.randn(1, q_heads, batch, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(1, kv_heads, batch, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(1, kv_heads, batch, head_dim, dtype=torch.float16, device="cuda")
    mod = SimpleNamespace(layer_idx=0, sliding_window=None)
    paged_backend_forward(mod, q, k, v, None, kv_context=ctx_obj)  # warm-up
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        paged_backend_forward(mod, q, k, v, None, kv_context=ctx_obj)
        sync()
    path = os.path.join(out_dir, f"backend_b{batch}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    kernels = sum(1 for e in events if e.get("ph") == "X" and e.get("cat") == "kernel")
    return {"batch": batch, "kernel_launches_per_call": kernels}


def hf_forward(ref, batch: int, prompt_len: int, n: int = 20) -> dict:
    from transformers import DynamicCache

    def prefill(b):
        ids = torch.randint(1, ref.config.vocab_size, (b, prompt_len), device="cuda")
        with torch.no_grad():
            out = ref(input_ids=ids, use_cache=True)
        return out.past_key_values

    res = {}
    for b in sorted({1, batch}):
        cache: DynamicCache = prefill(b)
        tok = torch.randint(1, ref.config.vocab_size, (b, 1), device="cuda")
        with torch.no_grad():
            for _ in range(3):
                ref(input_ids=tok, past_key_values=cache, use_cache=True)
            sync()
            ts = []
            for _ in range(n):
                t = PERF()
                ref(input_ids=tok, past_key_values=cache, use_cache=True)
                sync()
                ts.append((PERF() - t) * 1e3)
        res[f"batch_{b}_median_ms"] = median(ts)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profile_out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--out-len", type=int, default=256, help="output budget per request; must exceed the steps run on one engine")
    ap.add_argument("--ablations-only", action="store_true")
    ap.add_argument("--wall-only", action="store_true", help="median step time at batch 1, 8 and 32, nothing else")
    ap.add_argument("--per-sequence", action="store_true", help="old path: per-sequence attention (batched_decode=False)")
    args = ap.parse_args()
    global BATCHED_DECODE
    BATCHED_DECODE = not args.per_sequence
    os.makedirs(args.out, exist_ok=True)
    register_backend()
    results: dict = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch": args.batch,
                     "prompt_len": args.prompt_len}

    paged = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, attn_implementation=BACKEND_NAME).eval().to("cuda")
    cfg = paged.config
    B, P, O = args.batch, args.prompt_len, args.out_len

    if args.wall_only:
        rows = []
        for batch in (1, 8, 32):
            e = steady_engine(paged, cfg, num_blocks=4096, batch=batch, prompt_len=P, out_len=256)
            ts = wall_times(e, n=30)
            rows.append({"batch": batch, "median_ms": median(ts), "tok_per_s": batch / (median(ts) / 1e3)})
            del e
            torch.cuda.empty_cache()
        out = {"gpu": results["gpu"], "git": os.popen("git rev-parse --short HEAD").read().strip(),
               "batched_decode": BATCHED_DECODE, "rows": rows}
        with open(os.path.join(args.out, "wall_only.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(json.dumps(out, indent=2, default=float))
        return

    if args.ablations_only:
        out = {"gpu": results["gpu"], "batch": B, "ablations_ms": ablations(paged, cfg, B, P, 512)}
        with open(os.path.join(args.out, "ablations.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(json.dumps(out["ablations_ms"], indent=2, default=float))
        return

    eng = steady_engine(paged, cfg, num_blocks=4096, batch=B, prompt_len=P, out_len=O)
    results["wall_ms"] = wall_times(eng)
    results["wall_median_ms"] = median(results["wall_ms"])
    results["phases_ms"] = phase_times(eng)
    results["checks_off_path_ms"] = invariant_and_hash_cost(eng)
    results["cprofile_per_step"] = cprofile_step(eng)
    del eng
    torch.cuda.empty_cache()
    # 512 output tokens: the ablation runs about 380 steps on one engine.
    results["ablations_ms"] = ablations(paged, cfg, B, P, 512)
    eng = steady_engine(paged, cfg, num_blocks=4096, batch=B, prompt_len=P, out_len=O)
    results["trace"] = trace_step(eng, args.out)
    results["sync_sites"] = sync_sites(eng)
    results["isolated_backend_kernels"] = [isolated_backend(cfg.num_hidden_layers, b, args.out) for b in (1, 8, 32)]

    # Pool-size and batch/context scaling, each as a fresh engine.
    del eng
    torch.cuda.empty_cache()
    scaling = []
    for label, kw in [
        ("pool 4096, batch 16, ctx 64", dict(num_blocks=4096, batch=16, prompt_len=64, out_len=40)),
        ("pool 256, batch 16, ctx 64", dict(num_blocks=256, batch=16, prompt_len=64, out_len=40)),
        ("pool 4096, batch 8, ctx 64", dict(num_blocks=4096, batch=8, prompt_len=64, out_len=40)),
        ("pool 4096, batch 32, ctx 64", dict(num_blocks=4096, batch=32, prompt_len=64, out_len=40)),
        ("pool 4096, batch 32, ctx 256", dict(num_blocks=4096, batch=32, prompt_len=256, out_len=40)),
        ("pool 4096, batch 32, ctx 64, prefix cache on", dict(num_blocks=4096, batch=32, prompt_len=64, out_len=40, prefix=True)),
    ]:
        e = steady_engine(paged, cfg, **kw)
        ts = wall_times(e)
        scaling.append({"label": label, "median_ms": median(ts)})
        del e
        torch.cuda.empty_cache()
    results["scaling"] = scaling

    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).eval().to("cuda")
    results["hf"] = hf_forward(ref, B, P)
    results["ratio_engine_over_hf_batch"] = results["wall_median_ms"] / results["hf"][f"batch_{B}_median_ms"]
    results["ratio_engine_over_hf_batch1"] = results["wall_median_ms"] / results["hf"]["batch_1_median_ms"]

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in results.items() if k not in ("trace", "wall_ms")}, indent=2, default=float))
    print("wrote", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
