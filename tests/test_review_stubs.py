"""Review regression tests for the step 1 and step 2 implementations.

Each test asserts the correct behavior. A failure names the bug; nothing here fixes it.
"""

from __future__ import annotations

from engine_helpers import BLOCK, make_engine, make_models, prompts
from minivllm.clock import FakeClock
from minivllm.core.block_pool import BlockPool
from minivllm.core.invariants import check
from minivllm.core.prefix_cache import PrefixCache, block_hashes
from minivllm.core.sequence import SeqStatus


def cached_engine(device, num_blocks=8, max_batch=2):
    _, paged = make_models(device=device)
    # The per-step checker is off here: invariant 1 fails on any cached block (review #8) and would hide these bugs.
    engine = make_engine(
        paged, num_blocks=num_blocks, max_batch=max_batch, token_budget=64, prefix_cache=True, check_invariants=False
    )
    return engine


# review #1 (fixed): _admit_cached read self.tokens, which Scheduler never set, so every cached admission raised
def test_cached_admission_has_a_token_source(device):
    engine = cached_engine(device)
    engine.add_request(0, prompts([20])[0], 2)
    engine.run()
    assert engine.seqs[0].is_finished()


# review #2 (fixed): a successful admit that returns 0 (a miss) is read as a refusal, so the sequence stays waiting
# while its blocks stay allocated in PrefixCache.seq_blocks; the next admit then raises "already admitted"
def test_miss_is_admitted_not_refused(device):
    engine = cached_engine(device)
    engine.add_request(0, prompts([17])[0], 2)
    engine.step()
    assert engine.seqs[0].status == SeqStatus.RUNNING


# review #3 (fixed): PrefixCache.admit raises NoFreeBlocks when the pool is short; nothing catches it, so it escapes
# step(). Expected: admission stops (admit reports no room) and the request waits.
def test_pool_pressure_stalls_admission_instead_of_raising(device):
    engine = cached_engine(device, num_blocks=3, max_batch=3)
    for i, p in enumerate(prompts([17, 17, 17], seed=9)):
        engine.add_request(i, p, 4)
    outputs = engine.run(max_steps=500)
    assert set(outputs) == {0, 1, 2}


# review #4 (fixed): the BlockTable is never synced with the cache's blocks, cached_tokens adds prompt_len instead of
# the hit count, and cached_by_seq stores True instead of the count.
def test_cache_hit_syncs_table_and_counts_hit_tokens(device):
    engine = cached_engine(device)
    prompt = list(range(10, 42))  # 32 tokens: one cacheable block (the last token's block is recomputed)
    engine.add_request(0, prompt, 2)
    engine.run()
    engine.add_request(1, prompt, 2)
    engine.step()  # admits request 1 through the cache
    table = engine.scheduler.tables[1]
    assert table.num_tokens == len(prompt)
    assert list(table.block_ids) == engine.scheduler.prefix_cache.seq_blocks[1]
    assert engine.scheduler.cached_by_seq[1] == BLOCK
    assert engine.scheduler.cached_tokens == BLOCK


# review #5 (fixed): _free_blocks raises NotImplementedError whenever the cache is on, so every finish and every
# preemption crashes, and prefix_cache.free is never called.
def test_finished_sequence_is_released_through_the_cache(device):
    engine = cached_engine(device)
    engine.add_request(0, prompts([20])[0], 2)
    engine.run()
    assert engine.seqs[0].is_finished()
    assert engine.scheduler.prefix_cache.seq_blocks == {}


# review #6 (fixed): the idle jump does not release the request in the same call, so an idle gap costs one extra
# step() that schedules nothing. One request arriving at 5.0 with one output token should take exactly one step.
def test_idle_jump_releases_in_the_same_step(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    engine.add_request(0, prompts([3])[0], 1, arrival_time=5.0)
    calls = 0
    plain_step = engine.step

    def counting_step() -> list[int]:
        nonlocal calls
        calls += 1
        return plain_step()

    engine.step = counting_step
    engine.run()
    assert calls == 1


# review #7 (fixed): pending is a list in call order and only pending[0] is inspected. A request added with a later
# arrival first hides an earlier one, so the earlier request is scheduled late.
def test_pending_requests_are_released_in_arrival_order(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=2, token_budget=64, clock=FakeClock())
    p0, p1 = prompts([3, 3])
    engine.add_request(0, p0, 1, arrival_time=10.0)
    engine.add_request(1, p1, 1, arrival_time=5.0)
    engine.run()
    assert engine.logs[1].dequeue_time == 5.0
    assert engine.logs[0].dequeue_time == 10.0


# review #8 (fixed): check() counts only table-owned blocks, so any block the cache owns or keeps in its LRU
# breaks invariant 1 after a cached run.
def test_checker_accepts_cached_blocks_after_a_cached_run(device):
    engine = cached_engine(device)
    prompt = list(range(10, 42))
    engine.add_request(0, prompt, 2)
    engine.run()
    engine.add_request(1, prompt, 2)
    engine.run()
    check(engine.scheduler)


# review #9 (fixed): free() enqueues the chain root-first, so eviction removes the root first.
# A hit needs a contiguous chain from the root, so evicting the root orphans every cached descendant.
# Expected: the tail is evicted first and the root survives.
def test_eviction_keeps_the_root_of_a_cached_chain():
    pool = BlockPool(4)
    cache = PrefixCache(pool, BLOCK)
    chain = list(range(100, 148))  # 48 tokens: blocks 0 and 1 are cacheable
    cache.admit(0, chain)
    cache.mark_computed(0, len(chain))
    cache.free(0)
    cache.admit(1, list(range(200, 232)))  # 2 blocks: one free, then one eviction
    assert block_hashes(chain, BLOCK)[0] in cache.hash_to_block


# review #10 (fixed): hashes are registered at admit, before any KV exists. A sequence
# admitted and freed before its prefill runs leaves a hash on garbage KV, and a later admit hits it.
def test_freed_before_prefill_leaves_no_cache_hit():
    pool = BlockPool(4)
    cache = PrefixCache(pool, BLOCK)
    chain = list(range(100, 132))
    cache.admit(0, chain)
    cache.free(0)  # never computed
    assert cache.admit(1, chain) == 0


# not a bug: a hit on a refcount-0 block that is still in the LRU reuses it
def test_hit_on_lru_block_reuses_it():
    pool = BlockPool(4)
    cache = PrefixCache(pool, BLOCK)
    chain = list(range(100, 132))
    cache.admit(0, chain)
    cache.mark_computed(0, len(chain))
    cache.free(0)
    assert cache.admit(1, chain) == BLOCK  # 32 tokens: only the first block is cacheable
    assert all(block not in cache.lru for block in cache.seq_blocks[1])


# not a bug: the last prompt token is always recomputed, so a fully cached prompt hits one block fewer
def test_full_prompt_hit_recomputes_last_block():
    pool = BlockPool(4)
    cache = PrefixCache(pool, BLOCK)
    prompt = list(range(100, 132))
    cache.admit(0, prompt)
    cache.mark_computed(0, len(prompt))
    cache.free(0)
    assert cache.admit(1, prompt) == BLOCK


# not a bug: two requests with the same arrival time both run and finish
def test_same_arrival_time_is_fine(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=2, token_budget=64, clock=FakeClock())
    p0, p1 = prompts([3, 4])
    engine.add_request(0, p0, 1, arrival_time=5.0)
    engine.add_request(1, p1, 1, arrival_time=5.0)
    outputs = engine.run()
    assert set(outputs) == {0, 1}
    assert engine.logs[0].dequeue_time == engine.logs[1].dequeue_time == 5.0
