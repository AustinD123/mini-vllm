# Phase 3 — Prefix caching (problems 14–16)

Still on `FakeRunner`. Sequences now carry real token ids (the workload
generator can emit a shared system prompt + varied suffix).

---

## P14 — Chained block hashing
**File:** `minivllm/core/prefix_cache.py`
```python
def block_hashes(token_ids: list[int], block_size: int) -> list[int]: ...  # full blocks only
```
`hash_i = hash((hash_{i-1}, tuple(tokens in block i)))`.
**Tests to write:** two prompts sharing the first 40 tokens share exactly 2
block hashes (block 16); identical blocks at different prefixes hash
differently; the trailing partial block gets no hash.
**Checkpoint:** Why include the parent hash instead of hashing only the block's
tokens?
**vLLM ref:** `vllm/v1/core/kv_cache_utils.py` (block hash functions).

---

## P15 — Refcounted sharing
Extend the pool/manager: blocks have refcounts; a new sequence walks its
prompt hashes, reuses every hit (refcount++), allocates the rest; free
decrements and only returns a block at 0. `num_computed_tokens` starts at the
number of cached tokens.
**Tests to write:** two sequences sharing a prefix use fewer blocks than two
independent ones; freeing one leaves the shared blocks alive; a fully cached
prompt still recomputes its last token (needs logits).
**Checkpoint:** Why is copy-on-write unnecessary when only full blocks are
shared? Why must the last prompt token always be recomputed?

---

## P16 — LRU cache of freed blocks
Blocks hitting refcount 0 keep their hash and contents in an LRU; `allocate`
prefers truly free blocks, then evicts the least recently used cached block
(and removes its hash from the lookup table).
**Tests to write:** freed-then-requested prefix is a hit; under pressure the
oldest cached block is evicted first; an evicted hash is no longer a hit.
**Checkpoint:** How does this interact with recompute preemption, and why does
it make recompute cheap?
**vLLM ref:** `vllm/v1/core/block_pool.py` (cached block map, eviction).
