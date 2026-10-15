# Progress

Current problem: **P25**
Hardware: NVIDIA laptop GPU (VRAM: TBD) + free Colab T4

| # | Problem | Status |
|---|---|---|
| 1 | Block pool | reviewed |
| 2 | Slot mapping | reviewed |
| 3 | append_token / can_append | reviewed |
| 4 | Contiguous baseline + utilization | reviewed |
| 5 | Sequence state machine | reviewed |
| 6 | Static batching baseline | reviewed |
| 7 | FCFS continuous batching | reviewed |
| 8 | Token budget | reviewed |
| 9 | Chunked prefill | reviewed |
| 10 | Preemption (recompute) | reviewed |
| 11 | Cost model | reviewed |
| 12 | Metrics from event log | reviewed |
| 13 | Invariant checker + fuzz | reviewed |
| 14 | Chained block hashing | reviewed |
| 15 | Refcounted sharing | reviewed |
| 16 | LRU cached blocks | reviewed |
| 17 | Hand-written causal attention | reviewed |
| 18 | Hand-written decode loop | reviewed |
| 19 | KV pool + scatter | reviewed |
| 20 | Paged attention, one sequence | reviewed |
| 21 | Packed batch | reviewed |
| 22 | GQA | tests pass |
| 23 | HF backend + exactness | reviewed |
| 23b | Batched decode attention (agent-written) | tests pass |
| 24 | KV pool sizing | reviewed |
| 25 | Benchmark sweep | in progress |
| 26 | Fidelity check | todo |

Status values: todo · scaffolded · in progress · tests pass · reviewed (checkpoint answered)

## Engine and cache state (2026-10-15)

- Suite: 369 passed, 3 skipped (after P23b; the real-model P23 test passes with MINIVLLM_REAL_MODEL) (the env-gated real-model P23 test and the no-CUDA error-path test). The 10k-step cached fuzz is marked `slow`: `pytest -m "not slow"` skips it.
- All six re-review findings are fixed. Each is logged in `docs/fix-log.md` with its regression test.
- Prefix caching: a block's hash becomes a lookup entry only after its K/V is computed. Requests admitted in the same step cannot share blocks. On the 3050, prompt tokens forwarded fall by exactly the cached count (2,043 to 571). See `docs/fix-log.md` #9.
- KV metrics are split: "in use" (blocks held by running sequences) and "pool occupancy" (non-free, including the retained LRU). Plots use in use.
- Open: decode blocks are non-cacheable (differs from vLLM, see design log). Admission thrash: preemptions do not fall as the pool grows; the known mitigation is vLLM V0's watermark (design log).
- Decode step: one host-to-device copy per step, one sampling sync, and batched decode attention (P23b, agent-written). Step time at batch 32 is 1.7x an HF forward on the 3050; see `docs/perf-diagnosis.md` section 6 and `docs/fix-log.md` 10-14.
- Colab: `colab/mini_vllm_gpu_checks.ipynb` runs end to end locally in dry-run mode (twice, after the last change). The T4 run is not yet done. Its runtime estimate is 51 min with rates 0.5-4 req/s and sequential up to 0.5 req/s. The 0.25 req/s point is dropped to fit 60 min (see the config cell).
