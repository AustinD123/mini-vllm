# mini-vLLM — agent instructions

A learning project: a small LLM inference engine (paged KV cache, continuous
batching scheduler, prefix caching, paged attention on a real model). The human
is building it to *understand* it. Read `docs/plan.md` only if you need the big
picture.

## The one rule
The project is split into 26 numbered **problems** (specs in `docs/phase*.md`).
For a problem, the human writes the function body. You do not.
- You MAY write: stubs with signatures + docstrings, failing tests, fixtures,
  harness code, plotting, workload generator, server, HF loading, notebooks,
  config, anything not listed as a problem body.
- You MUST NOT write or paste the solution body of a problem, even partially,
  even "just to show", unless the human explicitly asks you to write that code
  in the current message. Without that explicit request, reply with one guiding
  question or hint first. Give a bigger hint only if asked again.
- Writing a solution body on request is allowed only when the human says so
  explicitly. Asking for help, being stuck, or saying "tell me" does not count.
- Reviewing is encouraged: point out bugs, missing edge cases, complexity issues.
  Describe the bug; let the human fix it.

## Workflow
- `/start-problem N` — scaffold problem N (skill in `.claude/skills/`).
- `/review-problem N` — review, quiz, compare with vLLM, log learnings.
- Current position lives in `docs/progress.md`. Read it at session start.
- Only open the phase doc for the current problem. Don't read the others.

## Layout
```
minivllm/core/      block_pool, block_table, sequence, scheduler, prefix cache
minivllm/runner/    base (ModelRunner interface), fake (cost model), real (HF + paged attn)
minivllm/attention/ paged attention backend
minivllm/bench/     workload gen, metrics, sweeps, plots
tests/              one test file per problem: tests/test_pNN_<name>.py
docs/               plan, phase specs, progress, design log
```

## Conventions
- Python 3.10+, pytest, type hints, no deps beyond torch/transformers/numpy/matplotlib.
- Block size 16 unless a test says otherwise.
- `num_computed_tokens` is the single source of truth for sequence progress.
- Seed every random workload. Benchmarks record GPU, versions, git commit.
- Keep the human's code style; don't refactor their solutions unasked.
