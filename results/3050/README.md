# RTX 3050 results (laptop, 4 GB)

Machine: NVIDIA GeForce RTX 3050 Laptop GPU, torch 2.14.1+cu130, transformers 5.19.0, Windows 11. Every JSON file has an `env` block with the GPU, versions and `git_commit`.

**Commit caveat.** These files were produced on `fe82a1e` with one uncommitted change: `RealRunner.forward(sample_rows=...)` and `Engine.step` (fix-log 15). The `git_commit` in each file therefore reads `fe82a1e`, not the commit that adds the change. `prefill_after_*.json` records the uncommitted files in `dirty_tracked_files`.

| File | What it is |
|---|---|
| `agreement_fp16.json`, `agreement_fp32.json` | Free-running agreement with HF `generate`, first divergence, and teacher-forced agreement, on 16 seeded requests (max_len 256) |
| `sizing_fp16.json` | KV pool blocks predicted from free memory against bytes measured on the GPU |
| `identical_outputs_gate.json` | Sequential, static and engine (batched and per-sequence) greedy outputs agree. Tiny float64 model, as in the notebook dry run |
| `prefix_diagnostics.json` | Prompt tokens forwarded with and without the prefix cache (FakeClock and WallClock) |
| `prefill_before_s{1..4}.json`, `prefill_after_s{1..4}.json` | Prefill step time and TTFT, four alternating serial sessions, before (`fe82a1e`) and after |

Not run here: the notebook's memory-pressure check (`agreement_pressure.json`) and the real-model float64 gate, which does not fit in 4 GB. Both run on the T4.
