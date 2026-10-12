"""Invariant checker: asserts the scheduler's bookkeeping is consistent.

Problem 13 — Invariant checker + fuzz

Story
    The scheduler moves blocks between the pool, the block tables, and the
    queues on every step, and a bug in any path can silently leak a block or
    lose a sequence. The checker runs after every step and fails loudly the
    moment the bookkeeping stops adding up, so a bug shows up at the step
    where it happens, not hundreds of steps later as a wrong latency number.

Operations
    check(scheduler) -> None
        asserts every invariant below. Returns None when all hold. Raises
        AssertionError with a message naming the broken invariant.

Invariants (checked together, after one step)
    1. Blocks add up: pool free count + blocks owned by all tables == total blocks.
    2. No block is owned twice: no block id appears in two tables, and no block
       id is both free and owned.
    3. Each running sequence has room for what it has computed: its table covers
       at least num_computed_tokens tokens.
    4. The queues are disjoint: no sequence is in both waiting and running, and
       a sequence is in waiting exactly when its status is WAITING or PREEMPTED.

Choices (the spec leaves these open)
    The checker may read a table's block ids and token count. If you would
    rather not reach into BlockTable's internals, add a read-only property on
    BlockTable and use that instead.
    An empty scheduler passes every invariant.

Example
    pool = BlockPool(4), sequence A holds 2 blocks
    pool.num_free == 2, owned == 2, total == 4    -> invariant 1 holds
    if block 2 is also listed in B's table        -> invariant 2 fails

Constraints
    check runs in time proportional to the number of blocks and sequences.
    check does not change the scheduler or the pool.

Edge cases to think about
    What does invariant 1 say about a block that was allocated and never given to a table?
    Can a preempted sequence still hold blocks? Should it?
    A sequence that just finished: which invariant covers its freed blocks?
    Which of these would catch the bug where a finished sequence keeps its table?

Tests
    tests/test_p13_invariants.py covers a healthy scheduler passing, a leaked
    block, a block owned twice, overlapping queues, and a seeded fuzz run that
    checks the invariants after every one of 10,000 steps.
"""

from __future__ import annotations

from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import SeqStatus


def check(scheduler: Scheduler) -> None:
    """Assert every scheduler invariant. Raises AssertionError when one is broken."""
    if not isinstance(scheduler, Scheduler):
        raise TypeError("scheduler must be a Scheduler instance")
    all_sequences = list(scheduler.running) + list(scheduler.waiting)
    if scheduler.pool.num_free + sum(len(table.block_ids) for table in scheduler.tables.values()) != scheduler.pool.num_blocks:
        raise AssertionError("Invariant 1 failed: blocks do not add up")
    all_owned_blocks = set()
    for table in scheduler.tables.values():
        for block_id in table.block_ids:
            if block_id in all_owned_blocks:
                raise AssertionError(f"Invariant 2 failed: block {block_id} is owned twice")
            all_owned_blocks.add(block_id)
    if not all_owned_blocks.isdisjoint(scheduler.pool.free_block_ids):
        raise AssertionError("Invariant 2 failed: a block is both free and owned")
    for sequence in all_sequences:
        if sequence.status == SeqStatus.RUNNING and sequence.num_computed_tokens > scheduler.tables[sequence.seq_id].num_tokens:
            raise AssertionError(f"Invariant 3 failed: sequence {sequence.seq_id} has computed more tokens than its table covers")
    waiting_ids = {seq.seq_id for seq in scheduler.waiting}
    running_ids = {seq.seq_id for seq in scheduler.running}
    for sequence in scheduler.waiting:
        if scheduler.tables[sequence.seq_id].block_ids:
            raise AssertionError(f"Invariant 4 failed: waiting sequence {sequence.seq_id} still holds blocks")
    if waiting_ids & running_ids:
        raise AssertionError("Invariant 4 failed: a sequence is both waiting and running")
    for sequence in all_sequences:
        if sequence.status in (SeqStatus.WAITING, SeqStatus.PREEMPTED) and sequence.seq_id not in waiting_ids:
            raise AssertionError(f"Invariant 4 failed: sequence {sequence.seq_id} is {sequence.status} but not in waiting")
        if sequence.status == SeqStatus.RUNNING and sequence.seq_id not in running_ids:
            raise AssertionError(f"Invariant 4 failed: sequence {sequence.seq_id} is RUNNING but not in running")   
        
