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

### P22 — GQA
Checkpoint answer: TODO (human to fill)

---

## P9 review findings (chunked prefill)

- **Partial chunks were never recorded.** `Scheduler.update` only advanced sequences in `sampled`, so a prefill chunk that didn't finish its prompt was lost and the next step re-requested it. Fixed: `update` now records progress for every sequence in the step's output, and `sampled` only decides whether that call samples.
- **The token budget wasn't used by the live scheduler.** `Scheduler.step` gave every running sequence all its uncomputed tokens. Fixed: `step` now builds its output from `plan_chunks`. Covered by `test_scheduler_splits_long_prompt_by_budget`.
- **Fully computed prefills were allocated and sampled.** Fixed: a prefill with nothing uncomputed is skipped.
- **A decode-list sequence with prompt left sampled early.** Fixed: only sequences with exactly 1 uncomputed token sample from the decode path.
- **Open: a sequence in both the decode and prefill lists** still counts against the budget twice (same as P8). Decide whether that's an error or silently deduplicated.
- **Open: running sequences don't reserve blocks for decode tokens.** Admission reserves the prompt only, so a long decode can outgrow its blocks. Needed before preemption (P10).

---

## P16 review notes (LRU cache of freed blocks)

- **Eviction is not O(1) in the worst case.** `_evict_least_recent` uses `next(iter(self.lru))` on a plain dict. Deleted entries stay at the front of the dict until it resizes, so repeated front evictions get slower as the cache grows. Measured: 5,000 evictions from 10,000 entries took 6 ms, and 50,000 evictions from 100,000 entries took 579 ms (about 10× per eviction for 10× the size). Open: pick a structure that removes the oldest entry in O(1).
- **A miss stops the hit search.** Blocks after an evicted block are recomputed even if they are still cached, because their KV depends on the evicted block.
- **Cached blocks break the P13 checker's invariant 1** once the scheduler uses the cache, since a refcount-zero cached block is neither free nor owned. Add the LRU to that count before the two are combined.

## Finding: repetition_penalty in HF generate under greedy decoding (2026-10-14)

## Decision: hash after compute, and the same-step sharing consequence (2026-10-14)
