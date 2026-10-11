# Design log

One entry per problem, written by the human (agent may tidy wording, not
content). This becomes the blog post.

## Template
### P<N> — <name> (<date>)
- **What I built:** 
- **What broke / what I got wrong first:** 
- **Checkpoint answer (my words):** 
- **How vLLM does it differently:** 
- **Open question:** 

---

---

## P9 review findings (chunked prefill)

- **Partial chunks were never recorded.** `Scheduler.update` only advanced sequences in `sampled`, so a prefill chunk that didn't finish its prompt was lost and the next step re-requested it. Fixed: `update` now records progress for every sequence in the step's output, and `sampled` only decides whether that call samples.
- **The token budget wasn't used by the live scheduler.** `Scheduler.step` gave every running sequence all its uncomputed tokens. Fixed: `step` now builds its output from `plan_chunks`. Covered by `test_scheduler_splits_long_prompt_by_budget`.
- **Fully computed prefills were allocated and sampled.** Fixed: a prefill with nothing uncomputed is skipped.
- **A decode-list sequence with prompt left sampled early.** Fixed: only sequences with exactly 1 uncomputed token sample from the decode path.
- **Open: a sequence in both the decode and prefill lists** still counts against the budget twice (same as P8). Decide whether that's an error or silently deduplicated.
- **Open: running sequences don't reserve blocks for decode tokens.** Admission reserves the prompt only, so a long decode can outgrow its blocks. Needed before preemption (P10).
