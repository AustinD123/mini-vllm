"""Review: the invariant checker and a 10k-step fuzz with prefix caching on.

The fuzz shares one token dict with the scheduler, the way the engine does.
"""

from __future__ import annotations

import random

import pytest

from minivllm.core.block_pool import BlockPool
from minivllm.core.invariants import check
from minivllm.core.prefix_cache import PrefixCache
from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import SeqStatus, Sequence

BLOCK = 16


# review #8 (fixed): check() counted only table-owned blocks, so blocks the cache owns broke invariant 1.
# A cached run must pass, including a block that has been released into the LRU.
def test_checker_counts_blocks_owned_by_the_cache():
    pool = BlockPool(8)
    cache = PrefixCache(pool, BLOCK)
    tokens = {0: list(range(100, 120))}  # 20 tokens: one full block, one partial
    sched = Scheduler(pool, BLOCK, max_batch=2, token_budget=64, prefix_cache=cache, tokens=tokens)
    seq = Sequence(seq_id=0, prompt_len=20, max_new_tokens=1, arrival_time=0.0)
    sched.add_request(seq)
    out = sched.step()  # admitted through the cache: the table and the cache both hold the blocks
    check(sched)
    sched.update(out, {0: 7})  # finishes; the full block is hashed, so it joins the LRU
    assert len(cache.lru) == 1
    check(sched)


def run_cached_fuzz(seed: int, steps: int, num_blocks: int, arrival_steps: int) -> int:
    """Staggered arrivals over a shared prefix; checks the invariants every step.

    Arrivals stop after arrival_steps, then the scheduler drains. Returns the number of finished requests.
    """
    rng = random.Random(seed)
    capacity = num_blocks * BLOCK
    shared = [rng.randint(1, 50) for _ in range(32)]
    pool = BlockPool(num_blocks)
    tokens: dict[int, list[int]] = {}
    sched = Scheduler(
        pool,
        BLOCK,
        max_batch=rng.randint(1, 4),
        token_budget=rng.randint(16, 128),
        prefix_cache=PrefixCache(pool, BLOCK),
        tokens=tokens,
    )
    all_seqs: dict[int, Sequence] = {}
    next_id = 0
    for step_no in range(steps):
        if step_no < arrival_steps and rng.random() < 0.3:
            tail = [rng.randint(1, 50) for _ in range(rng.randint(1, max(1, capacity // 4)))]
            prompt_ids = shared + tail
            max_new = rng.randint(1, max(1, capacity - len(prompt_ids)))
            if len(prompt_ids) + max_new > capacity:
                continue
            seq = Sequence(seq_id=next_id, prompt_len=len(prompt_ids), max_new_tokens=max_new, arrival_time=float(next_id))
            all_seqs[next_id] = seq
            tokens[next_id] = prompt_ids
            sched.add_request(seq)
            next_id += 1
        out = sched.step()
        # Sample what this step finishes computing, read after scheduling (admission can credit cache hits).
        sampled = {sid: 0 for sid, n in out.num_tokens.items() if n == all_seqs[sid].num_uncomputed()}
        sched.update(out, sampled)
        for sid in sampled:
            tokens[sid].append(0)  # the engine appends each sampled token; the cache reads it on re-admission
        check(sched)
    return sum(1 for s in all_seqs.values() if s.status == SeqStatus.FINISHED)


# review #8 (fixed): the 10k-step fuzz with prefix caching on. Every step passes the checker, and the run drains.
# Pool sizes are small enough to force preemption; the workload shares a 32-token prefix and arrives over time.
@pytest.mark.slow
@pytest.mark.parametrize("seed, num_blocks", [(1, 6), (2, 12), (3, 40)])
def test_cached_fuzz_holds_invariants_for_10k_steps(seed, num_blocks):
    finished = run_cached_fuzz(seed=seed, steps=10_000, num_blocks=num_blocks, arrival_steps=6_000)
    assert finished > 0
