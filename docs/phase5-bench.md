# Phase 5 — Benchmarks + validation (problems 25–26)

Agent scaffolds: sweep runner, Colab notebook, result JSON schema (GPU name,
torch/transformers versions, git commit, seed), plots.

---

## P25 — Benchmark sweep (T4, Qwen2.5-0.5B)
Systems: HF sequential, static batching, mini-vLLM (continuous + chunked
prefill), mini-vLLM + prefix caching.
Workload: Poisson arrivals, sweep λ until saturation; lognormal prompt/output
lengths fitted from public trace stats; `ignore_eos` so all systems emit the
same tokens; plus a shared-system-prompt workload for prefix caching.

The four headline numbers:
1. KV memory utilization, paged vs contiguous, same workload.
2. Throughput (output tok/s) vs static batching and sequential.
3. TTFT and TPOT p50/p99 vs request rate (latency from scheduled arrival).
4. Preemption rate vs KV pool size under memory pressure.

The human decides *what* to measure and interprets results; the agent writes
the plumbing.
**Checkpoint:** Why is your throughput number credible, and what would make it
not credible?

---

## P26 — Fidelity check
a) Run the same seeded workload through real vLLM on the T4 (fp16). Compare
   throughput and preemption behavior; explain differences.
b) Fit the P11 cost-model coefficients from RealRunner step timings; report
   simulator-predicted vs measured latency.
**Checkpoint:** Where does your engine diverge from vLLM and why (CUDA graphs,
fused kernels, scheduler policy details)?

vLLM files to cite in the write-up (checked against main @ 1dc60dd, 2026-10-09):
- `vllm/v1/core/sched/scheduler.py`, `sched/request_queue.py`
- `vllm/v1/core/kv_cache_manager.py`, `single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`, `kv_cache_utils.py`
