# Authorship

Who wrote which part of the code. The owner fills in the "Written by" column. The notes point at the
files in this repo that record each part.

| Component | Written by | Notes |
|---|---|---|
| `minivllm/core/`: block pool, block table, sequence, scheduler, static scheduler, prefix cache, contiguous baseline, invariants | | Problems P1 to P16 and P13 (invariants). Review fixes are in `docs/fix-log.md`. |
| `minivllm/engine.py` | | Step loop, arrival gate, prefix-cache admission (`8e51ca2`). One copy per step and batched sampling (`aba809b`). |
| `minivllm/clock.py` | | Fake and wall clocks. |
| `minivllm/runner/`: base, fake, real, hf_loop, packed, sizing | | Problems P17 to P24, runner side. `sizing.py` is P24. |
| `minivllm/attention/`: backend, paged, gqa, kv_pool, reference | | Problems P17 to P23. GQA is P22. |
| `minivllm/attention/batched.py` | | P23b. `docs/phase4-paged-attention.md` says "written by the agent". Wired in `4d809fd`. |
| `minivllm/bench/workload.py` | | `docs/phase2-scheduler.md` labels the workload generator "agent writes". Length constants are fitted by `scripts/fit_sharegpt.py` (`docs/sharegpt-fit.md`). |
| `minivllm/bench/`: baselines, sweep, plots, metrics, results, repeats, jobs, gpu_checks, fit_lengths | | `docs/phase5-bench.md` says "the human decides what to measure … the agent writes the plumbing". |
| `tests/` | | One file per problem, `tests/test_pNN_*.py`. Review regression tests are `tests/test_review_*.py`. |
| `scripts/` | | Profiling, diagnostics, GPU checks and the ShareGPT fit. |
| `colab/mini_vllm_gpu_checks.ipynb` | | Colab run of the checks and sweep. |
| `docs/design-log.md` | | The design-log file header says the human writes the entries. |
| `AGENTS.md`, `.claude/` | | Project instructions and skills. |
