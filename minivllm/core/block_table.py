"""Block table: maps token positions in one sequence to KV cache slots.

Problem 2 — Slot mapping

Story
    A sequence's tokens are stored in KV cache blocks, but the blocks a
    sequence owns are not next to each other in memory. The block table is
    the sequence's list of block ids, in the order the blocks were given to it.
    To write or read the keys and values of token number ``pos``, the engine
    needs a single flat slot number in the cache, so it knows exactly where
    that token lives.

Conventions
    block_table is the list of block ids for one sequence, in order.
    Token positions start at 0. Each block holds ``block_size`` slots.
    A slot is a flat index into the whole KV cache.

Operations
    slot_for_position(block_table, pos, block_size) -> int
        the slot that stores token ``pos``
    slot_mapping(block_table, start, end, block_size) -> list[int]
        the slots for positions start, start+1, ..., end-1 (end is exclusive)

Errors
    slot_for_position raises IndexError when pos is outside the range covered
    by the table, that is pos < 0 or pos >= len(block_table) * block_size.
    slot_mapping raises IndexError if any position in the range is outside
    that range. An empty range (start == end) returns [] and raises nothing.

Example
    block_table = [5, 2, 9], block_size = 16

    slot_for_position(block_table, 0, 16)    -> 80    (block 5, offset 0)
    slot_for_position(block_table, 15, 16)   -> 95    (block 5, offset 15)
    slot_for_position(block_table, 16, 16)   -> 32    (block 2, offset 0)
    slot_for_position(block_table, 37, 16)   -> 149   (block 9, offset 5)
    slot_for_position(block_table, 48, 16)   -> raises IndexError

    slot_mapping(block_table, 14, 18, 16)    -> [94, 95, 32, 33]
    slot_mapping(block_table, 3, 3, 16)      -> []

Constraints
    slot_for_position runs in O(1) time.
    slot_mapping runs in O(end - start) time.
    block_size >= 1. Block ids are non-negative integers.

Edge cases to think about
    What is the offset of the first slot in a block? The last?
    What happens to pos when block_size is 1?
    What is the first position that is out of range for a 3-block table?
    Why is the table indexed by block number and not by position?
    Which positions does the slot_mapping range include at its ends?

Tests
    tests/test_p02_slot_mapping.py covers the example above, a range that
    crosses two block boundaries, positions past the end of the table,
    and block_size 1.
"""

from __future__ import annotations


def slot_for_position(block_table: list[int], pos: int, block_size: int) -> int:
    """Return the KV cache slot that stores token ``pos``.

    Raises IndexError when ``pos`` is outside the range covered by the table.

    """
    if pos < 0 or pos >= len(block_table) * block_size:
        raise IndexError(f"pos {pos} is out of range for block_table of length {len(block_table)} and block_size {block_size}") 
    block_id=pos // block_size
    offset=pos % block_size
    return block_table[block_id] * block_size + offset
    raise NotImplementedError


def slot_mapping(
    block_table: list[int], start: int, end: int, block_size: int
) -> list[int]:
    """Return the slots for positions ``start`` up to but not including ``end``.

    Raises IndexError when any position in the range is outside the range
    covered by the table.
    """
    if start < 0 or end > len(block_table) * block_size:
        raise IndexError(f"start {start} or end {end} is out of range for block_table of length {len(block_table)} and block_size {block_size}")
    return [slot_for_position(block_table, pos, block_size) for pos in range(start, end)]

    raise NotImplementedError
