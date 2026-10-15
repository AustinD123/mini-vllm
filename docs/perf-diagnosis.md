# Decode-step performance diagnosis

Diagnosis only. No engine, scheduler or attention code was changed.

Written for: the project owner, who is building this to understand it.

## Setup

- GPU: NVIDIA GeForce RTX 3050 Laptop, 4 GB, driver 592.82, Windows 11
- torch 2.14.1+cu130, transformers 5.19.0
- Model: Qwen/Qwen2.5-0.5B-Instruct, fp16, paged backend (`minivllm_paged`)
- Steady decode: 32 running sequences, prompt 64 tokens, context about 70 to 110 during timing, pool of 4096 blocks, block size 16
- Scripts: `scripts/profile_step.py` (full profile and ablations), `scripts/profile_backend_isolated.py` (one backend call, no engine)
- Raw output: `profile_out/results.json`, `profile_out/ablations.json`, `profile_out/backend_isolated.txt` (the 747 MB Chrome trace is gitignored)

Run-to-run noise on this laptop is about ±15% to 20%. The same batch-32 config measured 536, 634 and 704 ms per step in different sessions. Read the ratios and the ablation deltas as the signal, and the absolute milliseconds as approximate.

## 1. Time breakdown per decode step (batch 32)

Phase timings from one run, CPU-side, with no extra syncs. Total is 648 ms. The unprofiled wall median in the same session was 536 ms, and the gap is session drift.

| Phase | ms per step | Share of engine step |
|---|---|---|
| scheduler.step | 0.08 | 0.01% |
| runner prep (positions, slot lists for all sequences, input tensors) | 0.32 | 0.05% |
| model, non-attention (embeddings, norms, linears, rope, lm_head) | 36.3 | 5.6% |
| paged backend, all 24 layers | 580.6 | 89.5% |
| sampling and bookkeeping (32 x `int(argmax)`, token append) | 31.2 | 4.8% |
| scheduler.update | 0.05 | 0.01% |
| **engine.step** | **648.5** | **100%** |

Checks that are off the benchmark path, measured for reference:

| Item | Cost | Runs per step? |
|---|---|---|
| `invariants.check` | 0.15 ms | No. Nothing outside the tests calls it. |
| Prefix re-hashing (`block_hashes`) | about 0.02 ms per prompt | Once per admission, not per step. Re-hashing all 32 prompts every step would cost 0.63 ms. |

Inside the backend, the 24 layers split as follows. Ablations are interleaved, median of 3 rounds of 40 steps, batch 32:

| Arm | Median ms per step | Change vs baseline |
|---|---|---|
| baseline | 663 | |
| slot host lists memoized across layers | 585 | -12% (slot Python, about 78 ms) |
| + slot device tensors memoized (removes the per-layer host-to-device copy and its sync) | 483 | -27% total (the copy and sync account for about 100 ms more) |
| `KVPool.write` without its two `torch.any` checks | 635 | -4%, within noise (ruled out) |

Isolated, one layer at batch 32, context 100, no engine:

| Mode | ms per call |
|---|---|
| enqueue only (no sync) | 17.4 |
| back to back, one sync at the end | 17.4 |
| synced after each call | 14.1 |

These times are the same whether or not the GPU is waited on, so the call is CPU-bound. Across 24 layers that is 340 to 420 ms, most of the step.

## 2. Questions

### a. Scaling

Each row is a separate engine, steady decode, median of 20 steps.

| Config | ms per step | tok/s (batch / step) |
|---|---|---|
| batch 8, pool 4096, ctx 64 | 220 | 36 |
| batch 16, pool 4096, ctx 64 | 349 | 46 |
| batch 32, pool 4096, ctx 64 | 704 | 45 |
| batch 32, pool 4096, ctx 256 | 759 | 42 |
| batch 32, pool 4096, ctx 64, prefix cache on | 690 | 46 |
| batch 16, **pool 256**, ctx 64 | 366 | 44 |
| batch 16, **pool 4096**, ctx 64 | 349 | 46 |

- **Batch: linear.** About 22 ms per sequence per step, so throughput stays flat at about 45 tok/s. That matches the flat 63 tok/s on the T4.
- **Total context: second order.** Quadrupling the context adds about 8%, which is inside the noise band.
- **Pool size: no effect.** 256 vs 4096 blocks gives 366 vs 349 ms. Step cost does not depend on `num_blocks`.

### b. Invariants and prefix hashing

Neither is on in the benchmark path. The invariant checker has no caller outside the tests. Prefix hashing runs in the `engine_prefix` system only, and only at admission. Measured costs are 0.15 ms (invariants, if enabled) and 0.02 ms per prompt (hashing, once per admission).

One caveat for the bench. `_admit_cached` retries `prefix_cache.admit` every step while the head request is blocked on blocks. Under pool pressure that re-hashes the same prompt each step. The steady-decode measurements here don't exercise that path.

### c. Kernels per paged attention call

Isolated, one call (one layer), counted from the trace:

| Batch | Kernels per call |
|---|---|
| 1 | 16 |
| 8 | 89 |
| 32 | 329 |

That is about 6 fixed kernels plus about 10 per sequence. Per sequence the calls are: slot tensor copy (memcpy), 2 index gathers (K and V), reshape, permute and copy, mask construction (`ones`, `tril`, `eq`, `masked_fill`), 2 einsums, divide, softmax, and the output `cat` and permute.

Per step, 24 layers at batch 32 is 24 x 329 = 7,896 kernels. The profiled step has 8,859 kernels in total, so the backend is 89% of launches. The other ~960 come from the HF model.

Syncs per step: 851 in total, from `sync-debug` attribution:

| Call site | Syncs per step | Cause |
|---|---|---|
| `minivllm/attention/backend.py:168` | 768 | `torch.tensor(slot list, device=cuda)`, once per sequence per layer |
| `minivllm/attention/kv_pool.py:85` | 48 | two `torch.any` range checks per layer |
| `minivllm/engine.py:192` | 32 | `int(logits[row].argmax())`, once per sequence |
| `minivllm/runner/real.py:46-48` | 3 | three input tensors |

### d. Python loops

- **Per sequence, per layer** (backend, 768 per step): slot list, host-to-device copy, two gathers, GQA call. This is the main loop.
- **Per token, per layer** (`block_table.slot_for_position`): 94,424 calls per step, recomputed in every layer for the same tokens.
- **Per sequence** (`engine.step` sampling loop): 32 syncs.
- **Per sequence** (runner prep): 32 `slot_mapping` calls, 0.3 ms total. Negligible.
- **Per layer** (backend validation): loops over 32 sequences, 24 times. Small.

No per-block Python loop is in the hot path. The per-token cost is inside `slot_for_position`.

## 3. Ranked causes

Ranked by measured cost.

1. **Per-sequence, per-layer loop in the paged backend, CPU-bound.** About 340 to 480 ms of the 536 ms step, from the isolated timing (17.4 ms per layer x 24) and from the backend phase.
   - Fix idea: one batched attention call per layer over all sequences, using an index tensor built once per step. That removes about 320 of the 329 launches per layer.
   - Expected saving: most of the ~480 ms. The step can't drop below the model's own forward, about 46 ms here. This is an estimate, not measured.

2. **Per-layer slot rebuild and host-to-device copy, with a sync each time.** 768 syncs per step, and per-token `slot_for_position` recomputed 24 times.
   - Measured by ablation: -27% (about 180 ms in that session), of which about 78 ms is the Python slot list and about 100 ms is the copy and its sync.
   - Fix idea: build the flat slot tensor once per step (`ctx.slots` already holds it) and make one host-to-device copy per step, not one per layer.
   - Expected saving: about 25% of the step. Measured by ablation, not an estimate. These two numbers overlap with cause 1, so the total gain is not additive.

3. **Per-sequence sampling syncs.** 32 `int(argmax)` calls, one sync each. Sampling and bookkeeping are 31 ms (4.8%).
   - Fix idea: one `argmax(-1)` over the sample rows and one `.tolist()` per step.
   - Expected saving: about 25 ms (4 to 5%). Estimate, not ablated.

Ruled out:

- `KVPool.write` checks: 48 syncs per step, but removing them changed nothing beyond noise (-4%). The GPU is idle anyway, so extra syncs cost little on their own.
- Invariant checking: off in the bench, 0.15 ms if enabled.
- Prefix re-hashing: off per step, 0.02 ms per admission.
- Pool size: no effect (256 vs 4096 blocks).
- Context length: about 8% for 4x context, within noise.
- Prefix cache: no effect on a workload with no shared prefixes.

## 4. Ratio against HF on the same batch

| Forward | Median ms |
|---|---|
| Engine decode step, batch 32, paged | 535.6 |
| HF forward, batch 32, contiguous DynamicCache, SDPA | 46.5 |
| HF forward, batch 1 | 45.7 |

- **Engine / HF at batch 32: 11.5x.** Engine / HF at batch 1: 11.7x.
- The T4 result was 6.5x (220 ms vs 34 ms).
- The gap reproduces, and it is larger here. The HF forward is 1.35x slower on the laptop (46 vs 34 ms). The engine is 2.4x slower (536 vs 220 ms). The engine's cost is CPU-side, and the laptop's host CPU hurts it more than it hurts HF. This is an inference from the numbers, not something I measured separately.

## 5. Caveats

- The profiled run is about 3.8x slower than the unprofiled one (1,892 vs 536 ms per step). Profiler numbers show where time goes in relative terms, but not the absolute split. The GPU-busy figure from the trace is about a quarter of the profiled step, with 76% idle. That agrees with the isolated timing (the GPU keeps up with the CPU), but I would not quote the 76% as a real idle fraction.
- cProfile inflates Python-heavy functions. The 62 ms cumulative for `slot_mapping` is an upper bound. The ablation (about 78 ms) is the better number.
- The ablations are measurement-only shims in `scripts/profile_step.py`: a copy of `KVPool.write` without its checks, a memo around `slot_mapping`, and a memo around `torch.tensor` inside the backend module. None of them is a proposed attention implementation.

## Reproduce

```
python scripts/profile_step.py --out profile_out                  # full profile, about 10 min
python scripts/profile_step.py --out profile_out --wall-only      # step time and HF ratio at batch 1, 8, 32
python scripts/profile_step.py --out profile_out --wall-only --per-sequence   # old attention path
python scripts/profile_backend_isolated.py                        # one backend call, no engine
```

The ablation arms in the "before" section were removed from the script after the fixes, because they patched code paths that no longer exist.

## 6. After the fixes (docs/fix-log.md 10-14)

Every row is measured in one session on the same machine, with HF timed alongside, so the ratios are comparable. Absolute times drift between sessions (HF measured 46 ms in the first session and 83–95 ms in this one), so read the ratio columns.

| Commit | What changed | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|---|
| cd06f22 (before) | none | 137.7 ms, 1.66x HF | 373.3 ms, 4.18x | 1184.5 ms, 12.43x |
| aba809b | per-step tensors and one copy (fix 10), batched sampling (fix 12), no pool checks (fix 11) | 130.9 ms, 1.59x | 350.7 ms, 3.81x | 1023.1 ms, 9.09x |
| 4d809fd, per-sequence path (`--per-sequence`) | same as aba809b | 119.4 ms, 1.29x | 346.7 ms, 4.22x | 998.9 ms, 11.16x |
| 4d809fd, batched decode (default) | batched decode attention (fix 13) | 139.2 ms, 1.53x | 148.1 ms, 1.67x | 152.5 ms, 1.72x |

Throughput (tok/s): before 7.3, 21.4, 27.0 at batch 1, 8, 32. After: 7.2, 54.0, 209.8.

Per step, batch 32, from the full profile of 4d809fd (`profile_after1/`):

| Measure | Before (cd06f22) | After (4d809fd) |
|---|---|---|
| Synchronizing calls per step | 851 | 3 |
| Kernel launches per step (profiled) | 8,859 | 1,485 |
| Paged backend calls per step | 24 | 24 (one batched call each) |
| Backend CPU time per step | 580.6 ms | 66.3 ms |
| Model non-attention per step | 36.3 ms | 82.1 ms (same session, larger share of a smaller total) |
| Sampling and bookkeeping | 31.2 ms | 4.7 ms |

The "model non-attention" figure rose because the rest of the step got much smaller. The HF-equivalent work did not change.

**Targets.** Within 2x of an HF forward at batch 32: met (1.72x wall, 1.3x in the profile session). Throughput rising with batch size: met (7, 54, 210 tok/s). No extra profiling round was needed.

**What is left.** At batch 1 the engine is 1.5x HF, with about 50 ms of fixed per-step cost on this laptop. Per layer the backend still launches about 20 small kernels of its own (validation-free gather, pool write index math, the decode mask). The next candidate is hoisting the decode mask and the write indices to once per step. Not done this round: the gain is within the noise at batch 32.

**Sync fixes alone are small.** Removing 848 of 851 syncs took batch 32 from 1184 to 1023 ms (-14%). The ablation in section 1 predicted about -27%. The sync stall was real, but most of the cost was per-sequence launches, which only the batched call removes.
