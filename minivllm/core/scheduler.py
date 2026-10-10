"""FCFS continuous batching: admit and retire requests every step.

Problem 7 — FCFS continuous batching step

Story
    Static batching makes short requests wait for the longest one in their
    batch. A continuous scheduler instead re-decides the batch on every step.
    When a request finishes, its blocks go back to the pool and a waiting request
    can take its place on the very next step. The scheduler decides who runs
    each step, and the runner computes it.

Operations
    Scheduler(pool, block_size, max_batch, token_budget)
                                         pool: the shared BlockPool; max_batch: most
                                         sequences running at once; token_budget is
                                         stored but not used until problem 8
    add_request(seq) -> None             queue a request behind the waiting ones
    step() -> SchedulerOutput            admit what fits, then return every running
                                         sequence's uncomputed tokens for this step
    update(output, sampled) -> list      apply a step's results; return finished
                                         sequences and free their blocks

Choices (the spec leaves these open)
    Each sequence owns a BlockTable (problem 3). The scheduler keeps the
    seq_id -> BlockTable mapping itself.
    Order within step(): running sequences grow their blocks first, then
    waiting requests are admitted.
    Admission is strict FCFS: stop at the first waiting request that does not
    fit. Do not skip ahead to a smaller one.
    A request fits when its block table can append its whole uncomputed prompt
    and the batch has a free slot. The check is all-or-nothing.
    A running sequence that cannot get a block for its tokens raises NoFreeBlocks.
    Preemption, which handles that case, is problem 10.
    An admitted sequence becomes RUNNING. A finished one becomes FINISHED.

Errors
    Scheduler(...) raises ValueError when max_batch < 1.
    NoFreeBlocks propagates from step() if a running sequence cannot grow.

Example (block_size 16, pool = BlockPool(4), max_batch 2)
    A: prompt 20, max_new 2     needs 2 blocks
    B: prompt 5,  max_new 1     needs 1 block
    C: prompt 3,  max_new 1     needs 1 block

    add A, B, C
    step 1: admit A (2 blocks), admit B (1 block). C waits: batch is full.
            output {A: 20, B: 5}.   pool.num_free -> 1
            update(sampled {A, B}): A has 1 output, B has 1 output and finishes.
            B's block is freed.     pool.num_free -> 2
    step 2: A is running and needs 1 more token. It fits in its second block.
            admit C (1 block). output {A: 1, C: 3}.   pool.num_free -> 1
            update(sampled {A, C}): A has 2 outputs and finishes (frees 2 blocks).
            C has 1 output and finishes (frees 1 block).   pool.num_free -> 4

Constraints
    step() and update() run in time proportional to the number of running
    sequences plus the number admitted or finished this step.
    A request that arrives mid-run is first scheduled on the next step() call.

Edge cases to think about
    A request that arrives while a batch is running: when does it first compute?
    A big request at the head of the queue and a small one behind it: what should
    happen to the small one, and why?
    When a sequence finishes, when does its block become available to a waiting request?
    What should step() return when nothing is waiting and nothing is running?
    A sequence is FINISHED after update. Can it still appear in a later output?

Tests
    tests/test_p07_scheduler.py covers a request joining mid-run, strict FCFS
    behind a request that doesn't fit, blocks freed when a sequence finishes,
    and the max_batch limit.
"""

from __future__ import annotations

import queue

from minivllm.core.block_pool import BlockPool
from minivllm.core.block_table import BlockTable
from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.core.sequence import SeqStatus, Sequence


class Scheduler:
    """Continuous batching scheduler. See the module docstring for the contract."""

    def __init__(self, pool: BlockPool, block_size: int, max_batch: int, token_budget: int) -> None:
        """Create a scheduler over ``pool``. Raises ValueError when max_batch < 1."""
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.pool = pool
        self.block_size = block_size
        self.max_batch = max_batch
        self.token_budget = token_budget
        self.waiting: queue.Queue[Sequence] = queue.Queue()
        self.running: list[Sequence] = []
        self.tables: dict[int, BlockTable] = {}

    def add_request(self, seq: Sequence) -> None:
        """Queue ``seq`` behind every request already waiting."""
        self.tables[seq.seq_id] = BlockTable(self.pool, self.block_size)
        self.waiting.put(seq)

    def step(self) -> SchedulerOutput:
        """Grow running sequences, admit waiting ones in FCFS order, and return the step's work."""
        while len(self.running) < self.max_batch and not self.waiting.empty():
            seq = self.waiting.queue[0]  # peek at the first waiting sequence
            table = self.tables[seq.seq_id]
            if not table.can_append(seq.num_uncomputed()):
                break  # stop admitting if the first one doesn't fit
            table.append_tokens(seq.num_uncomputed())
            self.running.append(self.waiting.get())
            seq.set_status(SeqStatus.RUNNING)
        output = {seq.seq_id: seq.num_uncomputed() for seq in self.running}
        return SchedulerOutput(output)


    def update(self, output: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]:
        """Apply one step's results. Finished sequences free their blocks and are returned."""
        for seq_id in sampled:
            if seq_id not in output.num_tokens:
                raise ValueError(f"Sequence {seq_id} in sampled but not in output")
        finished_sequences = []
        still_running = []
        for seq in self.running:
            if seq.seq_id in sampled:
                seq.on_tokens_computed(output.num_tokens[seq.seq_id], True)
            if seq.is_finished():
                seq.set_status(SeqStatus.FINISHED)
                finished_sequences.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running
        for seq in finished_sequences:
            self.tables.pop(seq.seq_id).free_all()
        return finished_sequences
