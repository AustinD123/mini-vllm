"""Sequence state machine: one request's progress through the engine.

Problem 5 — Sequence state machine

Story
    Every request moves through the engine: it waits for admission, runs,
    can be kicked out to make room for others, and eventually finishes. The
    scheduler decides what each step does by asking each sequence how many of
    its tokens still need computing, and whether it is still in prefill. This
    class keeps those answers consistent as the sequence moves through states.

Operations
    Sequence(seq_id, prompt_len, max_new_tokens, arrival_time)
                                       a new sequence, status WAITING, nothing computed
    num_uncomputed() -> int            prompt + output tokens not yet computed
    is_prefill() -> bool               True while the prompt is not fully computed
    on_tokens_computed(n, sampled)     record n tokens computed; sampled=True means
                                       this step also produced a new output token
    preempt_recompute() -> None        throw away computed state, keep the outputs
    is_finished() -> bool              True once output_len reaches max_new_tokens
    set_status(new_status) -> None     move to a new SeqStatus (checked, see below)

Legal status transitions (a choice: the spec lists the states but not the moves)
    WAITING   -> RUNNING
    RUNNING   -> PREEMPTED, FINISHED
    PREEMPTED -> RUNNING
    Anything else raises ValueError. preempt_recompute does not change status;
    the scheduler calls set_status(PREEMPTED) itself.

Errors
    on_tokens_computed raises ValueError when n < 0 or n > num_uncomputed().
    on_tokens_computed raises ValueError when sampled=True and n is not all the
        uncomputed tokens (a choice: a sample only happens once the step has
        caught up with everything it was given).
    set_status raises ValueError on an illegal transition.

Example
    seq = Sequence(seq_id=0, prompt_len=4, max_new_tokens=2, arrival_time=0.0)
    seq.num_uncomputed()            -> 4
    seq.is_prefill()                -> True
    seq.on_tokens_computed(4, True) -> num_computed 4, output_len 1
    seq.num_uncomputed()            -> 1
    seq.is_prefill()                -> False
    seq.on_tokens_computed(1, True) -> num_computed 5, output_len 2
    seq.is_finished()               -> True
    seq.preempt_recompute()         -> num_computed 0, output_len still 2
    seq.num_uncomputed()            -> 6
    seq.is_prefill()                -> True

    seq.set_status(RUNNING)         -> ok (from WAITING)
    seq.set_status(FINISHED)        -> ok (from RUNNING)
    seq.set_status(RUNNING)         -> raises ValueError

Constraints
    Every method runs in O(1) time.
    prompt_len >= 1 and max_new_tokens >= 1.
    num_computed_tokens is the only record of progress: num_uncomputed and
    is_prefill are derived from it, never stored separately.

Edge cases to think about
    What is num_uncomputed right after the prompt is computed, before any output?
    Is a decode step's sampled token counted as computed in that same step?
    What should a chunk of the prompt that doesn't finish it do with sampled?
    Should on_tokens_computed be allowed on a FINISHED sequence?
    Why does preempt_recompute keep output_len but reset num_computed_tokens?

Tests
    tests/test_p05_sequence.py covers an illegal transition, num_uncomputed
    after preemption, a single decode step, and finishing at max_new_tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SeqStatus(Enum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED = 4


@dataclass
class Sequence:
    """One request's progress. See the module docstring for the contract."""

    seq_id: int
    prompt_len: int
    max_new_tokens: int
    arrival_time: float
    num_computed_tokens: int = 0
    output_len: int = 0
    status: SeqStatus = SeqStatus.WAITING

    def num_uncomputed(self) -> int:
        """Prompt plus output tokens that have not been computed yet."""
        return self.prompt_len + self.output_len - self.num_computed_tokens


    def is_prefill(self) -> bool:
        """True while the prompt is not fully computed."""
        return self.num_computed_tokens < self.prompt_len
    
    def on_tokens_computed(self, n: int, sampled: bool) -> None:
        """Record ``n`` computed tokens; ``sampled`` means a new output token.

        Raises ValueError for an invalid n, or sampled=True on a partial step.
        """

        if n < 0 or n > self.num_uncomputed():
            raise ValueError("Invalid number of computed tokens")
        if sampled and n != self.num_uncomputed():
            raise ValueError("sampled=True only allowed for a single computed token")
        if sampled and self.output_len >= self.max_new_tokens:
            raise ValueError("Sequence already produced max_new_tokens outputs")
        self.num_computed_tokens += n
        if sampled:
            self.output_len += 1

    def preempt_recompute(self) -> None:
        """Reset computed progress to zero, keeping the outputs produced so far."""
        self.set_status(SeqStatus.PREEMPTED)
        self.num_computed_tokens = 0

    def is_finished(self) -> bool:
        """True once output_len has reached max_new_tokens."""
        return self.output_len >= self.max_new_tokens
        

    def set_status(self, new_status: SeqStatus) -> None:
        """Move to ``new_status`` if the transition is legal.

        Raises ValueError on an illegal transition.
        """
        legal_transitions = {
            SeqStatus.WAITING: {SeqStatus.RUNNING},
            SeqStatus.RUNNING: {SeqStatus.PREEMPTED, SeqStatus.FINISHED},
            SeqStatus.PREEMPTED: {SeqStatus.RUNNING},
        }
        if new_status not in legal_transitions.get(self.status, set()):
            raise ValueError("Illegal status transition")
        self.status = new_status
