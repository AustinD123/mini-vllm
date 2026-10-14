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

from collections import deque

from minivllm.core.block_pool import BlockPool
from minivllm.core.block_table import BlockTable
from minivllm.core.prefix_cache import PrefixCache
from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.core.sequence import SeqStatus, Sequence


class Scheduler:
    """Continuous batching scheduler. See the module docstring for the contract."""

    def __init__(
        self,
        pool: BlockPool,
        block_size: int,
        max_batch: int,
        token_budget: int,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        """Create a scheduler over ``pool``. Raises ValueError when max_batch < 1."""
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.pool = pool
        self.block_size = block_size
        self.max_batch = max_batch
        self.token_budget = token_budget
        self.prefix_cache = prefix_cache
        self.cached_tokens = 0
        # Cached prompt tokens at each sequence's most recent admission.
        self.cached_by_seq: dict[int, int] = {}
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.tables: dict[int, BlockTable] = {}

    def add_request(self, seq: Sequence) -> None:
        """Queue ``seq`` behind every request already waiting."""
        capacity = self.pool.num_blocks * self.block_size
        if seq.prompt_len + seq.max_new_tokens > capacity:
            raise ValueError(f"Request {seq.seq_id} cannot fit in the pool")
        self.tables[seq.seq_id] = BlockTable(self.pool, self.block_size)
        self.waiting.append(seq)

    def _admit(self) -> None:
        """Admit waiting sequences in FCFS order while a slot and room for the prompt exist."""
        while len(self.running) < self.max_batch and self.waiting:
            seq = self.waiting[0]
            if self.prefix_cache is not None:
                if not self._admit_cached(seq):
                    break
                continue
            table = self.tables[seq.seq_id]
            if not table.can_append(seq.num_uncomputed()):
                break
            table.append_tokens(seq.num_uncomputed())
            self.waiting.popleft()
            self.running.append(seq)
            seq.set_status(SeqStatus.RUNNING)

    def _admit_cached(self, seq: Sequence) -> bool:
        """Admit the head of the waiting queue through the PrefixCache (step 1, human-written).

        Called from _admit in place of the BlockTable path when prefix_cache is set.
        Returns False when the request does not fit; _admit then stops (strict FCFS).
        Returns True after the sequence is admitted. Admission must:
            - ask prefix_cache.admit(seq.seq_id, prompt tokens) for the cached prefix
            - record cached_by_seq[seq.seq_id] and add it to cached_tokens
            - pop the sequence from waiting, append it to running, set it RUNNING
        The sequence's BlockTable must end up holding the same block ids the
        cache handed out, because the engine reads slots from the table.
        """
        if not self.prefix_cache.admit(seq.seq_id, self.tokens[seq.seq_id][:seq.prompt_len]):
            return False
        self.cached_by_seq[seq.seq_id] = True
        self.cached_tokens += seq.prompt_len
        self.waiting.popleft()
        self.running.append(seq)
        seq.set_status(SeqStatus.RUNNING)
        return True

    def _free_blocks(self, seq_id: int) -> None:
        """Release a sequence's blocks. Without a prefix cache, this frees its table (step 1, human-written with caching on)."""
        if self.prefix_cache is not None:
            raise NotImplementedError("step 1: release through self.prefix_cache.free")
        self.tables[seq_id].free_all()

    def _preempt(self, victim: Sequence) -> None:
        """Free the victim's blocks and put it at the front of the waiting queue."""
        self.running.remove(victim)
        self._free_blocks(victim.seq_id)
        victim.preempt_recompute()
        self.waiting.appendleft(victim)

    def step(self) -> SchedulerOutput:
        """Admit, plan this step's tokens, grow blocks (preempting if needed), and return the work."""
        self._admit()
        decoding = [seq for seq in self.running if seq.num_uncomputed() == 1]
        prefilling = [seq for seq in self.running if seq.num_uncomputed() > 1]
        allocation, _ = plan_chunks(decoding, prefilling, self.token_budget)
        for seq in list(self.running):
            n = allocation.get(seq.seq_id, 0)
            if n == 0:
                continue
            table = self.tables[seq.seq_id]
            # room needed beyond what the table already holds for this sequence
            needed = seq.num_computed_tokens + n - table.num_tokens
            while needed > 0 and not table.can_append(needed):
                victim = choose_victim(self.running)
                self._preempt(victim)
                allocation.pop(victim.seq_id, None)
                if victim is seq:
                    break
            if seq.status == SeqStatus.RUNNING and needed > 0:
                table.append_tokens(needed)
        return SchedulerOutput(allocation)

    def update(self, output: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]:
        """Apply one step's results. Finished sequences free their blocks and are returned."""
        for seq_id in sampled:
            if seq_id not in output.num_tokens:
                raise ValueError(f"Sequence {seq_id} in sampled but not in output")
        finished_sequences = []
        still_running = []
        for seq in self.running:
            if seq.seq_id in output.num_tokens:
                seq.on_tokens_computed(output.num_tokens[seq.seq_id], seq.seq_id in sampled)
            if seq.is_finished():
                seq.set_status(SeqStatus.FINISHED)
                finished_sequences.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running
        for seq in finished_sequences:
            self._free_blocks(seq.seq_id)
            self.tables.pop(seq.seq_id)
        return finished_sequences


def split_budget(decoding: list[Sequence], prefilling: list[Sequence], token_budget: int) -> dict[int, int]:
    """Split one step's token budget between decodes and prefills.

    Problem 8 — Per-step token budget

    Story
        A step can only do so much work before its latency becomes too high. The
        budget caps the tokens computed in one step. Decodes come first because
        each running request is waiting on exactly one token, so they are cheap
        and every one of them matters for latency. Prefills use whatever is left.

    Operations
        split_budget(decoding, prefilling, token_budget) -> dict[seq_id, tokens]
            decoding: running sequences that need one token each this step
            prefilling: waiting sequences whose prompt is not computed yet, in
                        arrival order
            returns the tokens each scheduled sequence computes this step

    Choices (the spec leaves these open)
        Each decode takes 1 token. Decodes are taken in list order until the
        budget runs out. Any decode that doesn't fit this step is left out.
        Prefills get the remaining budget, in list order, and only whole
        prompts: a prefill gets its full uncomputed tokens or nothing.
        Strict FCFS: once a prefill doesn't fit, stop. Do not skip ahead to a
        shorter one. (Splitting a long prompt across steps is problem 9.)
        token_budget == 0 schedules nothing.

    Errors
        Raises ValueError when token_budget < 0.

    Example (token_budget = 64)
        decoding:   ten sequences ids 0..9, each with 1 uncomputed token
        prefilling: id 10 with 54 uncomputed, id 11 with 10 uncomputed

        decodes take 10 -> 54 left
        id 10 needs 54 -> fits -> 0 left
        id 11 needs 10 -> does not fit, nothing left
        result: {0: 1, ..., 9: 1, 10: 54}

        token_budget = 4 with the same ten decodes
        result: {0: 1, 1: 1, 2: 1, 3: 1}   (no prefill)

    Constraints
        Runs in time proportional to the number of sequences passed in.
        Does not change any sequence's state.

    Edge cases to think about
        What happens to the prefill when the budget runs out exactly at a decode?
        Can a decode be scheduled with more than 1 token?
        A prefill of exactly the remaining budget: does it fit?
        Why must a prefill that doesn't fit block the ones behind it?

    Tests
        tests/test_p08_token_budget.py covers the 64-budget split, a budget
        smaller than the number of decodes, strict FCFS for prefills, and budget 0.
    """
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    allocation: dict[int, int] = {}
    candidates = decoding + prefilling
    decodes = [seq for seq in candidates if seq.num_uncomputed() == 1]
    prefills = [seq for seq in candidates if seq.num_uncomputed() > 1]
    for seq in decodes:
        if token_budget <= 0:
            break
        allocation[seq.seq_id] = 1
        token_budget -= 1
    for seq in prefills:
        if token_budget <= 0:
            break
        needed = seq.num_uncomputed()
        if needed <= token_budget:
            allocation[seq.seq_id] = needed
            token_budget -= needed
        else:
            break  # strict FCFS: stop if the first prefill doesn't fit
    return allocation


def plan_chunks(
    decoding: list[Sequence], prefilling: list[Sequence], token_budget: int
) -> tuple[dict[int, int], list[int]]:
    """Plan one step with chunked prefill. Returns (allocation, sample_ids).

    Problem 9 — Chunked prefill

    Story
        A long prompt, such as 4000 tokens, can take many steps of compute. If
        it had to fit in one step, it would hold up every running request for
        that whole step. Chunking lets a prompt advance a slice at a time, so
        decodes keep getting their one token per step and the prompt still
        finishes.

    Operations
        plan_chunks(decoding, prefilling, token_budget) -> (allocation, sample_ids)
            allocation: seq_id -> tokens computed this step
            sample_ids: seq_ids that produce an output token this step

    Choices (the spec leaves these open)
        Decodes are planned first, as in problem 8. Each takes 1 token.
        Prefills take what is left, in arrival order. A prefill may take a
        partial chunk: min(uncomputed, remaining budget). Once the budget hits
        0, stop.
        A decode always samples, because its one token is its next output.
        A prefill samples only when its chunk covers all of its uncomputed
        tokens. A partial chunk does not sample.
        token_budget == 0 schedules nothing.

    Errors
        Raises ValueError when token_budget < 0.

    Example (token_budget = 256)
        No decodes, one prompt of 1000 tokens (id 0):
        step 1: {0: 256}                 samples: none
        step 2: {0: 256}                 samples: none
        step 3: {0: 256}                 samples: none
        step 4: {0: 232}                 samples: [0]   (chunk completes the prompt)

        One decode (id 1) running next to the same 1000-token prompt (id 0):
        step 1: {1: 1, 0: 255}           samples: [1]
        step 2: {1: 1, 0: 255}           samples: [1]
        step 3: {1: 1, 0: 255}           samples: [1]
        step 4: {1: 1, 0: 235}           samples: [1, 0]

    Constraints
        Runs in time proportional to the number of sequences passed in.
        Does not change any sequence's state.

    Edge cases to think about
        What is num_uncomputed for a prompt that is already partly computed?
        A prompt of exactly the remaining budget: is it a full chunk, and does it sample?
        Why must a sample only happen on the chunk that finishes the prompt?
        What do the chunks add up to, and how would you check that?

    Tests
        tests/test_p09_chunked_prefill.py covers the 1000-token prompt split into
        four steps, samples only on the last chunk, a decode that is never starved
        alongside a long prompt, and budget boundaries.
    """
    if token_budget < 0:
        raise ValueError("Token budget cannot be negative.")
    allocation: dict[int, int] = {}
    sample_ids: list[int] = []
    for seq in decoding:
        if token_budget <= 0:
            break
        allocation[seq.seq_id] = 1
        token_budget -= 1
        if seq.num_uncomputed() == 1:
            sample_ids.append(seq.seq_id)
    for seq in prefilling:
        if token_budget <= 0:
            break
        needed = seq.num_uncomputed()
        chunk = min(needed, token_budget)
        if chunk > 0:
            allocation[seq.seq_id] = chunk
            token_budget -= chunk
            if chunk == needed:
                sample_ids.append(seq.seq_id)
    return allocation, sample_ids


def choose_victim(running: list[Sequence]) -> Sequence:
    """Pick the sequence to preempt when a running sequence cannot get a block.

    Problem 10 — Preemption (recompute)

    Story
        The pool can run out mid-sequence. Decodes keep needing one more block
        as they grow, and a finished request only frees its blocks when it
        ends. Something has to give. The engine preempts one running sequence:
        it frees that sequence's blocks so the others can keep growing, and the
        victim recomputes its state later. Its outputs are kept, so no output
        text is lost.

    Operations
        choose_victim(running) -> Sequence
            returns the most recently arrived sequence in ``running``

    Choices (the spec leaves these open)
        "Newest" means the largest arrival_time. Ties are not expected.
        The scheduler, not this function, frees the victim's blocks, sets it
        to PREEMPTED, and puts it at the front of the waiting queue.

    Errors
        Raises ValueError when ``running`` is empty.

    Example
        running: A (arrival 0.0), B (arrival 1.0)
        choose_victim([A, B]) -> B
        choose_victim([]) -> raises ValueError

    Constraints
        Runs in O(n) time over the running sequences. Does not change any sequence.

    Edge cases to think about
        What should happen when there is only one running sequence?
        Why preempt the newest rather than the oldest? What does the oldest already have?
        Could the victim be the sequence that needs the block?

    Tests
        tests/test_p10_preemption.py covers the newest being the victim, and the
        scheduler preempting under a small pool, with every request still finishing,
        rejecting a request that can never fit, and not admitting new requests
        while a preempted one waits.
    """
    if not running: 
        raise ValueError("No running sequences to choose from.")
    return max(running, key=lambda seq: seq.arrival_time)

