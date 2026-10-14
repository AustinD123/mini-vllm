"""Regression tests for the six re-review fixes. Each test fails on the code as it was before its fix."""

from __future__ import annotations

import math

import pytest

from engine_helpers import BLOCK, make_engine, make_models, prompts
from minivllm.clock import FakeClock
from minivllm.core.block_pool import BlockPool
from minivllm.core.block_table import BlockTable
from minivllm.core.invariants import check
from minivllm.core.prefix_cache import PrefixCache
from minivllm.core.scheduler import Scheduler, choose_victim
from minivllm.core.sequence import SeqStatus, Sequence


# review: fix 1. An oversized request raises at the call site and leaves nothing in pending, seqs or logs.
def test_oversized_request_is_rejected_before_any_state_changes(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=2, max_batch=1, token_budget=64, clock=FakeClock())
    with pytest.raises(ValueError):
        engine.add_request(0, prompts([20])[0], 20)  # 40 tokens; the pool holds 32
    assert 0 not in engine.seqs and 0 not in engine.tokens and 0 not in engine.logs
    assert engine.pending == []
    assert not engine.has_work()


# review: fix 1 (empty prompt and zero output are also refused up front)
def test_empty_prompt_and_zero_output_are_refused(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    with pytest.raises(ValueError):
        engine.add_request(0, [], 3)
    with pytest.raises(ValueError):
        engine.add_request(1, prompts([3])[0], 0)
    assert engine.pending == [] and engine.seqs == {}


# review: fix 2. A request whose arrival has already passed is in the scheduler before the next step.
def test_past_arrival_added_mid_run_is_visible_to_the_scheduler_at_once(device):
    _, paged = make_models(device=device)
    clock = FakeClock()
    engine = make_engine(paged, num_blocks=8, max_batch=2, token_budget=64, clock=clock)
    engine.add_request(0, prompts([3])[0], 2)
    engine.step()
    clock.advance_to(10.0)
    engine.add_request(1, prompts([3])[0], 2, arrival_time=5.0)  # the arrival time is already in the past
    queued = [s.seq_id for s in engine.scheduler.waiting] + [s.seq_id for s in engine.scheduler.running]
    assert 1 in queued


# review: fix 3. The checker rejects a pending hash whose sequence is not running.
# The cache holds no blocks for sequence 99, so only the pending check can catch this entry.
def test_checker_rejects_pending_hashes_of_a_sequence_that_is_not_running():
    pool = BlockPool(8)
    cache = PrefixCache(pool, BLOCK)
    tokens = {0: list(range(100, 120))}
    sched = Scheduler(pool, BLOCK, max_batch=2, token_budget=64, prefix_cache=cache, tokens=tokens)
    sched.add_request(Sequence(seq_id=0, prompt_len=20, max_new_tokens=1, arrival_time=0.0))
    sched.step()
    check(sched)
    cache.pending[99] = [(0, 5, 111)]  # a stale pending entry for a sequence that never ran
    with pytest.raises(AssertionError, match="Invariant 4"):
        check(sched)


# review: fix 3. The checker rejects hash maps that are not exact inverses.
def test_checker_rejects_hash_maps_that_are_not_inverses():
    pool = BlockPool(8)
    cache = PrefixCache(pool, BLOCK)
    sched = Scheduler(pool, BLOCK, max_batch=2, token_budget=64, prefix_cache=cache, tokens={})
    cache.hash_to_block[12345] = 3  # no matching block_to_hash entry
    with pytest.raises(AssertionError, match="Invariant 2"):
        check(sched)


# review: fix 4. If adopt fails after the cache admitted the sequence, the cache admit is rolled back.
def test_failed_adopt_rolls_back_the_cache_admit(monkeypatch):
    pool = BlockPool(8)
    cache = PrefixCache(pool, BLOCK)
    tokens: dict[int, list[int]] = {}
    sched = Scheduler(pool, BLOCK, max_batch=2, token_budget=64, prefix_cache=cache, tokens=tokens)
    shared = list(range(100, 140))  # 40 tokens: two full blocks become cacheable
    tokens[0] = shared
    sched.add_request(Sequence(seq_id=0, prompt_len=40, max_new_tokens=1, arrival_time=0.0))
    out = sched.step()
    sched.update(out, {0: 7})  # prefill done: the blocks are hashed, and sequence 0 finishes
    tokens[1] = list(shared)
    sched.add_request(Sequence(seq_id=1, prompt_len=40, max_new_tokens=1, arrival_time=1.0))
    free_before = pool.num_free
    lru_before = list(cache.lru)

    def broken_adopt(self, block_ids, num_tokens):
        raise RuntimeError("adopt failed")

    monkeypatch.setattr(BlockTable, "adopt", broken_adopt)
    with pytest.raises(RuntimeError):
        sched.step()
    assert 1 not in cache.seq_blocks
    assert pool.num_free == free_before
    assert list(cache.lru) == lru_before
    assert [s.seq_id for s in sched.waiting] == [1]
    assert sched.waiting[0].status == SeqStatus.WAITING
    check(sched)


# review: fix 5. A NaN or infinite arrival time is refused before any state changes.
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_arrival_time_is_refused(device, bad):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, clock=FakeClock())
    with pytest.raises(ValueError):
        engine.add_request(0, prompts([3])[0], 2, arrival_time=bad)
    assert engine.pending == [] and engine.seqs == {}


# review: fix 6. The victim is the newest by (arrival, seq_id), whatever the order of the running list.
def test_choose_victim_breaks_arrival_ties_on_seq_id():
    seqs = [Sequence(seq_id=i, prompt_len=1, max_new_tokens=1, arrival_time=0.0) for i in (3, 1, 2)]
    assert choose_victim(seqs).seq_id == 3
    assert choose_victim(list(reversed(seqs))).seq_id == 3
    assert choose_victim(seqs[1:]).seq_id == 2  # ids 1 and 2 remain: 2 is newer
