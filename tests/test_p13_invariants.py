import random

import pytest

from minivllm.core.block_pool import BlockPool
from minivllm.core.invariants import check
from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import SeqStatus, Sequence

BLOCK_SIZE = 16


def make(seq_id: int, prompt_len: int, max_new: int) -> Sequence:
    return Sequence(seq_id=seq_id, prompt_len=prompt_len, max_new_tokens=max_new, arrival_time=float(seq_id))


def healthy_scheduler() -> Scheduler:
    sched = Scheduler(BlockPool(8), BLOCK_SIZE, max_batch=2, token_budget=1000)
    sched.add_request(make(0, 20, 50))
    sched.add_request(make(1, 4, 50))
    for _ in range(3):
        out = sched.step()
        sched.update(out, {sid: 0 for sid in out.num_tokens})
    return sched


def test_healthy_scheduler_passes_check():
    check(healthy_scheduler())


# extra edge case
def test_empty_scheduler_passes_check():
    check(Scheduler(BlockPool(4), BLOCK_SIZE, max_batch=2, token_budget=100))


def test_leaked_block_is_detected():
    sched = healthy_scheduler()
    sched.pool.allocate()  # taken from the pool but owned by no table
    with pytest.raises(AssertionError):
        check(sched)


def test_block_owned_twice_is_detected():
    sched = healthy_scheduler()
    owner = next(t for t in sched.tables.values() if t._blocks)
    other = next(t for t in sched.tables.values() if t is not owner)
    other._blocks.append(owner._blocks[0])
    with pytest.raises(AssertionError):
        check(sched)


def test_sequence_in_both_waiting_and_running_is_detected():
    sched = healthy_scheduler()
    sched.waiting.append(sched.running[0])
    with pytest.raises(AssertionError):
        check(sched)


def run_fuzz(seed: int, steps: int) -> None:
    """Random arrivals, random lengths, a small pool; check the invariants every step."""
    rng = random.Random(seed)
    num_blocks = rng.randint(4, 12)
    capacity = num_blocks * BLOCK_SIZE
    sched = Scheduler(
        BlockPool(num_blocks),
        BLOCK_SIZE,
        max_batch=rng.randint(1, 4),
        token_budget=rng.randint(16, 128),
    )
    all_seqs: dict[int, Sequence] = {}
    next_id = 0
    for _ in range(steps):
        if rng.random() < 0.3:
            prompt = rng.randint(1, capacity // 2)
            max_new = rng.randint(1, capacity - prompt)
            seq = make(next_id, prompt, max_new)
            all_seqs[next_id] = seq
            sched.add_request(seq)
            next_id += 1
        before = {sid: all_seqs[sid].num_uncomputed() for sid in all_seqs}
        out = sched.step()
        sampled = {sid: 0 for sid, n in out.num_tokens.items() if n == before[sid]}
        sched.update(out, sampled)
        check(sched)


def test_fuzz_invariants_hold_every_step():
    for seed in range(10):
        run_fuzz(seed, steps=1000)


# review: a sequence waiting (not running) must not hold blocks; preemption frees them
def test_waiting_sequence_holding_blocks_is_detected():
    sched = healthy_scheduler()
    waiting_seq = make(9, 8, 1)
    sched.add_request(waiting_seq)
    sched.tables[9].append_tokens(8)
    with pytest.raises(AssertionError):
        check(sched)


# review: the fuzzer must drain every request, not just pass the checker
def test_fuzz_drains_every_request():
    rng = random.Random(3)
    num_blocks = 6
    sched = Scheduler(BlockPool(num_blocks), BLOCK_SIZE, max_batch=3, token_budget=64)
    seqs = [make(i, rng.randint(1, 60), rng.randint(1, 30)) for i in range(20)]
    for seq in seqs:
        sched.add_request(seq)
    for _ in range(5000):
        if all(s.status == SeqStatus.FINISHED for s in seqs):
            break
        before = {s.seq_id: s.num_uncomputed() for s in seqs}
        out = sched.step()
        sampled = {sid: 0 for sid, n in out.num_tokens.items() if n == before[sid]}
        sched.update(out, sampled)
        check(sched)
    assert all(s.status == SeqStatus.FINISHED for s in seqs)
