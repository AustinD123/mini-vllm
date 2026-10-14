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
| 24 | KV pool sizing | reviewed |
| 25 | Benchmark sweep | in progress |
| 26 | Fidelity check | todo |

Status values: todo · scaffolded · in progress · tests pass · reviewed (checkpoint answered)

## Engine and cache state (2026-10-14)

- Suite: 320 passed before the prefix-cache review, plus the new sweep and benchmark tests (see the commit that adds them). Cached-fuzz tests run 10k steps each (three seed and pool pairs), about two minutes.
- Prefix caching: a block's hash becomes a lookup entry only after its K/V is computed (`PrefixCache.mark_computed`). Requests admitted in the same step cannot share blocks. The shared-prefix benchmark uses Poisson arrivals so sharing is measurable.
- Open, from the re-review: decode blocks are never registered, so generated text is not cacheable. `add_request` validates capacity only at release, not at call time. `invariants.check` does not verify pending entries or that the hash maps are inverses.
- Real model (Qwen2.5-0.5B-Instruct): P23 real-model test passes. fp16 agreement is 99.4% teacher-forced; fp32 agrees on every position.
- Colab: `colab/mini_vllm_gpu_checks.ipynb` runs end to end locally in dry-run mode. The T4 run is not yet done.
