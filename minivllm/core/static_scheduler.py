"""Static batching baseline: run a batch to completion, then take the next one.

Problem 6 — Static batching baseline

Story
    The simplest way to batch requests is to take a group of waiting requests,
    run them together, and only start the next group once every request in
    the current one has finished. Short requests sit idle while the longest
    request in their batch keeps going. This is the baseline that continuous
    batching is measured against.

Operations
    StaticScheduler(batch_size)        B = the most requests in one batch
    add_request(seq) -> None           queue a request (arrival order)
    step() -> SchedulerOutput          work for this step: each running
                                       sequence's uncomputed tokens
    update(output, sampled) -> list    apply a step's results. sampled maps
                                       seq_id -> the token produced this step.
                                       Returns the sequences that finished.

Step rules (choices, since the spec doesn't pin them)
    A new batch is taken only when no batch is running. It is the first
    min(B, waiting) requests in arrival order. Admitted sequences move to RUNNING.
    Each step, every running sequence computes all of its uncomputed tokens.
    That means the whole prompt on the first step. Prefill is not chunked here
    (chunking is problem 9).
    A sequence that samples in a step gets exactly one output that step.
    When a sequence reaches max_new_tokens, update marks it FINISHED.
    A batch ends when all of its sequences have finished.

Errors
    add_request raises ValueError when batch_size < 1.
    update raises ValueError if a sequence in sampled is not in the output.

Example (B = 2)
    sched = StaticScheduler(batch_size=2)
    sched.add_request(seq 0: prompt 4, max_new 2)
    sched.add_request(seq 1: prompt 4, max_new 3)
    sched.add_request(seq 2: prompt 4, max_new 1)

    step 1: step() -> {0: 4, 1: 4}      seq 2 waits
            update(sampled={0, 1})      seq 0 has 1 output, seq 1 has 1 output
    step 2: step() -> {0: 1, 1: 1}
            update(sampled={0, 1})      seq 0 finishes (2 outputs)
    step 3: step() -> {1: 1}            seq 1 still running; batch not done
            update(sampled={1})         seq 1 finishes (3 outputs)
    step 4: step() -> {2: 4}            the batch is empty, so seq 2 starts

Constraints
    step() and update() cost time proportional to the batch size.
    batch_size >= 1. A request is never admitted into a batch that is already running.

Edge cases to think about
    What should step() return when nothing is waiting and nothing is running?
    If the batch is 2 long and 2 short requests, when does the short one
    finish, and when does the next batch start?
    A request with max_new_tokens = 1: how many steps does it occupy?
    Should a finished request in a running batch be scheduled again?

Tests
    tests/test_p06_static_batching.py covers batch length set by the longest
    request ([10, 200] runs for 200 steps), no admission while a batch runs,
    and batch size limits.
"""

from __future__ import annotations

from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.core.sequence import SeqStatus, Sequence


class StaticScheduler:
    """Batches requests statically. See the module docstring for the contract."""

    def __init__(self, batch_size: int) -> None:
        """Create a scheduler that runs at most ``batch_size`` requests at once."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        """Queue ``seq`` behind every request already waiting.

        Raises ValueError when batch_size < 1.
        """
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.waiting.append(seq)    
        

    def step(self) -> SchedulerOutput:
        """Return the uncomputed tokens of every running sequence for this step."""
        if not self.running:
            # Start a new batch if there are waiting sequences
            self.running = self.waiting[:self.batch_size]
            self.waiting = self.waiting[self.batch_size:]
            for seq in self.running:
                seq.set_status(SeqStatus.RUNNING)
        output = {seq.seq_id: seq.num_uncomputed() for seq in self.running}
        return SchedulerOutput(output)
        

    def update(self, output: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]:
        """Apply one step's results and return the sequences that finished.

        Raises ValueError if a sequence in ``sampled`` is not in ``output``.
        """
        for s in sampled:
            if s not in output.num_tokens:
                raise ValueError(f"Sequence {s} in sampled but not in output")
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
        return finished_sequences       
