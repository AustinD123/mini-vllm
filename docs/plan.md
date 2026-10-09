# Plan

## Goal
Understand scheduling and memory management for LLM serving by building them,
and produce benchmark numbers defensible in interviews. Deliverable: repo + blog
post. Reference bar (not a template to copy): PageServe
(github.com/TryingtobeingNikhil/vLLM_Inference_Engine).

## Architecture
One interface decouples scheduling from execution:

```
Scheduler.step() -> SchedulerOutput  (which seqs, how many tokens each)
ModelRunner.execute(SchedulerOutput) -> sampled tokens + step latency
```

- `FakeRunner` (phases 1–3): no model, latency from a cost model, random
  tokens. Runs on a laptop in ms; used to hammer the scheduler.
- `RealRunner` (phase 4): Qwen2.5-0.5B with our own paged attention, on a
  local NVIDIA GPU or Colab T4. Produces the published numbers.

Scheduler bugs and attention bugs never appear at the same time.

## Phases
| Phase | Doc | Problems | Est. hours |
|---|---|---|---|
| 1 Memory | phase1-memory.md | 1–5 | ~8 |
| 2 Scheduler | phase2-scheduler.md | 6–13 | ~14 |
| 3 Prefix caching | phase3-prefix.md | 14–16 | ~7 |
| 4 Real model + paged attention | phase4-paged-attention.md | 17–24 | ~22 |
| 5 Benchmarks + validation | phase5-bench.md | 25–26 | ~10 |

~60–65 h total ≈ 11–12 weeks at 5–6 h/week. Showable checkpoint after
phase 2 (~week 4): simulator + blog post part 1.

## Out of v1 (stretch)
CPU swap preemption, n-gram speculative decoding, streaming/OpenAI server,
C++ port of the block allocator via pybind11 (only worth it with a measured
overhead win).

## Learnings
Every problem ends with an entry in `docs/design-log.md`: what I built, what
broke, how vLLM differs. The log becomes the blog post.
