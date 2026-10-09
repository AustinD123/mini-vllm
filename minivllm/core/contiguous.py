"""Contiguous baseline: reserve max_len slots per sequence up front.

Problem 4 — Contiguous baseline + utilization

Story
    The simplest KV cache gives every sequence one contiguous region sized for
    the longest sequence the engine accepts (max_len). Nobody knows a
    sequence's final length in advance, so each one reserves max_len slots at
    admission, even if it only ends up using a few. This class models that
    policy so we can measure how much memory it wastes compared with paged
    blocks.

Operations
    ContiguousAllocator(total_slots, max_len)  pool of total_slots, each sequence reserves max_len
    can_admit() -> bool                        True if one more sequence's reservation fits
    admit(seq_id) -> None                      reserve max_len slots for seq_id
    append(seq_id, n) -> None                  record n more tokens for seq_id
    release(seq_id) -> None                    free seq_id's reservation
    utilization(used_slots, reserved_slots) -> float
                                               fraction of reserved slots actually used

Errors
    admit(seq_id) raises NoSlotsAvailable when can_admit() is False.
    admit(seq_id) raises ValueError when seq_id is already admitted.
    append(seq_id, n) raises KeyError when seq_id is not admitted.
    append(seq_id, n) raises ValueError when n < 0, or when the sequence would
        exceed max_len tokens. A failed append changes nothing.
    release(seq_id) raises KeyError when seq_id is not admitted.
    utilization raises ValueError when used_slots < 0, reserved_slots < 0, or
        used_slots > reserved_slots.
    utilization returns 0.0 when reserved_slots is 0 (a choice: an empty
        reservation counts as zero use, so benchmarks never divide by zero).

Example
    alloc = ContiguousAllocator(total_slots=4096, max_len=1024)
    alloc.can_admit()          -> True
    alloc.admit(0)             -> reserves 1024 slots
    alloc.append(0, 10)        -> the sequence has 10 tokens; its reservation is still 1024
    alloc.admit(1)
    alloc.admit(2)
    alloc.admit(3)             -> four reservations now fill 4096 slots
    alloc.can_admit()          -> False
    alloc.admit(4)             -> raises NoSlotsAvailable
    alloc.release(2)
    alloc.can_admit()          -> True

    utilization(60, 3072)      -> 60 / 3072 = 0.01953125
    utilization(60, 80)        -> 0.75

Constraints
    admit, append, release and can_admit run in O(1) time.
    total_slots >= 0 and max_len >= 1.
    If total_slots < max_len, the allocator never admits anything.

Edge cases to think about
    A sequence that appends exactly max_len tokens: is that allowed?
    What happens to can_admit after a release of a sequence that never appended?
    Does append need to know anything about other sequences?
    What is the utilization of a sequence that finished after 20 tokens with max_len 1024?

Tests
    tests/test_p04_contiguous.py covers the four-sequence limit for 4096 slots
    and max_len 1024, utilization for a fixed length list against hand-computed
    values, and release behavior.
"""

from __future__ import annotations


class NoSlotsAvailable(Exception):
    """Raised by ContiguousAllocator.admit when no full reservation fits."""


class ContiguousAllocator:
    """Reserves ``max_len`` slots per admitted sequence out of ``total_slots``.

    See the module docstring for the full problem statement.
    """

    def __init__(self, total_slots: int, max_len: int) -> None:
        """Create an allocator with no sequences admitted."""
        if total_slots < 0:
            raise ValueError("total_slots must be non-negative")
        if max_len < 1:
            raise ValueError("max_len must be at least 1")
        self.total_slots = total_slots
        self.max_len = max_len
        self.admitted = {}  # seq_id -> used_slots
        self.slots_used = 0

    def can_admit(self) -> bool:
        """True if one more max_len reservation fits in the remaining slots."""
        if(self.total_slots - self.slots_used >= self.max_len):
            return True
        return False
        raise NotImplementedError

    def admit(self, seq_id: int) -> None:
        """Reserve max_len slots for ``seq_id``.

        Raises NoSlotsAvailable when no reservation fits, and ValueError when
        ``seq_id`` is already admitted.
        """
        if seq_id==None:
            raise ValueError("seq_id cannot be None")
        if seq_id in self.admitted:
            raise ValueError("Sequence already admitted")
        if not self.can_admit():
            raise NoSlotsAvailable("No slots available for reservation")
        self.admitted[seq_id] = 0
        self.slots_used += self.max_len

    def append(self, seq_id: int, n: int) -> None:
        """Record ``n`` more tokens for ``seq_id``.

        Raises KeyError for an unknown sequence, and ValueError when n is
        negative or the sequence would exceed max_len. A failed append changes
        nothing.
        """
        if seq_id not in self.admitted:
            raise KeyError("Sequence not admitted")
        if n < 0:   
            raise ValueError("Cannot append negative number of tokens")
        if n + self.admitted[seq_id] > self.max_len:
            raise ValueError("Appending would exceed max_len")
        self.admitted[seq_id] += n 


    def release(self, seq_id: int) -> None:
        """Free the reservation held by ``seq_id``.

        Raises KeyError when ``seq_id`` is not admitted.
        """
        if seq_id not in self.admitted:
            raise KeyError("Sequence not admitted")
        del self.admitted[seq_id]
        self.slots_used -= self.max_len


def utilization(used_slots: int, reserved_slots: int) -> float:
    """Return used_slots / reserved_slots, or 0.0 when reserved_slots is 0.

    Raises ValueError for negative inputs or used_slots > reserved_slots.
    """
    if used_slots < 0 or reserved_slots < 0:
        raise ValueError("used_slots and reserved_slots must be non-negative")
    if used_slots > reserved_slots:
        raise ValueError("used_slots cannot exceed reserved_slots")
    if reserved_slots == 0:
        return 0.0
    return used_slots / reserved_slots
    raise NotImplementedError
