"""Review: the invariant checker and a 10k-step fuzz with prefix caching on.

The scheduler has no token source for cached admission (review #1), so the fuzz attaches
``scheduler.tokens`` by hand, the same contract the cached admission path reads.
"""

from __future__ import annotations

import random

from minivllm.core.block_pool import BlockPool
from minivllm.core.invariants import check
from minivllm.core.prefix_cache import PrefixCache
from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import SeqStatus, Sequence

BLOCK = 16


# review #8 (bug): check() counts only table-owned blocks. A block the cache owns (admitted, not yet in a table)
# breaks invariant 1 even though nothing is wrong. Built without the scheduler's admission path.
def test_checker_counts_blocks_owned_by_the_cache():
    pool = BlockPool(8)
    cache = PrefixCache(pool, BLOCK)
    sched = Scheduler(pool, BLOCK, max_batch=2, token_budget=64, prefix_cache=cache)
    cache.admit(0, list(range(100, 120)))  # two blocks held by the cache for sequence 0
    check(sched)


def run_cached_fuzz(seed: int, steps: int) -> int:
    """Random arrivals over a shared prefix; checks the invariants every step. Returns the number of finished requests."""
    rng = random.Random(seed)
    num_blocks = rng.randint(6, 12)
    capacity = num_blocks * BLOCK
    shared = [rng.randint(1, 50) for _ in range(32)]
    pool = BlockPool(num_blocks)
    sched = Scheduler(
        pool,
        BLOCK,
        max_batch=rng.randint(1, 4),
        token_budget=rng.randint(16, 128),
        prefix_cache=PrefixCache(pool, BLOCK),
    )
    sched.tokens = {}  # review #1 workaround: the token source the cached admission path expects
    all_seqs: dict[int, Sequence] = {}
    next_id = 0
    for _ in range(steps):
        if rng.random() < 0.3:
            tail = [rng.randint(1, 50) for _ in range(rng.randint(1, max(1, capacity // 4)))]
            prompt_ids = shared + tail
            max_new = rng.randint(1, max(1, capacity - len(prompt_ids)))
            if len(prompt_ids) + max_new > capacity:
                continue
            seq = Sequence(seq_id=next_id, prompt_len=len(prompt_ids), max_new_tokens=max_new, arrival_time=float(next_id))
            all_seqs[next_id] = seq
            sched.tokens[next_id] = prompt_ids
            sched.add_request(seq)
            next_id += 1
        before = {sid: s.num_uncomputed() for sid, s in all_seqs.items()}
        out = sched.step()
        sampled = {sid: 0 for sid, n in out.num_tokens.items() if n == before[sid]}
        sched.update(out, sampled)
        check(sched)
    return sum(1 for s in all_seqs.values() if s.status == SeqStatus.FINISHED)


# review #8 (bug): the 10k-step fuzz with prefix caching on. Expected: every step passes the checker and requests finish.
def test_cached_fuzz_holds_invariants_for_10k_steps():
    finished = run_cached_fuzz(seed=11, steps=10_000)
    assert finished > 0
