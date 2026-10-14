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

from minivllm.core.block_pool import BlockPool, NoFreeBlocks


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


class BlockTable:
    """Problem 3 — append_token / can_append

    Story
        One sequence grows one token at a time during decoding, and all at once
        during prefill. Its tokens live in blocks taken from the shared pool.
        The table has to hand out new blocks as the sequence grows, and it has
        to refuse an append cleanly when the pool is too full, so the scheduler
        can decide what to do next without the table being left half-updated.

    Operations
        BlockTable(pool, block_size)       an empty table that uses blocks from pool
        num_blocks_needed(n) -> int        how many new blocks appending n tokens would take
        can_append(n) -> bool              True if the pool can cover appending n tokens
        append_tokens(n) -> None           add n tokens, allocating blocks as needed
        free_all() -> None                 give every block back to the pool, reset to empty
        num_tokens -> int                  tokens stored so far (property)

    Errors
        append_tokens(n) raises NoFreeBlocks when the pool cannot cover all n tokens.
        In that case nothing changes: no block is allocated, num_tokens is unchanged.
        A negative n raises ValueError in every method that takes a count.

    Example (block_size = 16, pool = BlockPool(3))
        bt = BlockTable(pool, 16)
        bt.num_tokens              -> 0
        bt.num_blocks_needed(1)    -> 1
        bt.append_tokens(16)       -> pool.num_free drops by 1 (one full block)
        bt.num_tokens              -> 16
        bt.num_blocks_needed(1)    -> 1
        bt.append_tokens(1)        -> pool.num_free drops by 1 more
        bt.num_tokens              -> 17
        bt.can_append(32)          -> False  (49 tokens need 4 blocks; 2 held, 1 free)
        bt.append_tokens(32)       -> raises NoFreeBlocks
        bt.num_tokens              -> 17     (unchanged)
        bt.free_all()              -> pool.num_free is back to 3
        bt.num_tokens              -> 0

    Constraints
        num_tokens and can_append run in O(1) time.
        append_tokens runs in time proportional to the number of blocks it allocates.
        A table never holds a block id it has not been given by the pool.

    Edge cases to think about
        A table with 15 tokens and block size 16: how many blocks does one more token need?
        What should append_tokens(0) do to the pool?
        If the pool has exactly enough blocks, does can_append say yes?
        If append_tokens fails halfway through, what state is left behind, and how would you avoid it?
        After free_all, can the table be used again straight away?

    Tests
        tests/test_p03_append_tokens.py covers one-token appends at the block
        boundaries, a 40-token append, the all-or-nothing rule, and free_all.
    """

    def __init__(self, pool: BlockPool, block_size: int) -> None:
        """Create an empty table that takes blocks from ``pool``."""
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.pool = pool
        self.block_size = block_size
        self._blocks: list[int] = []
        self._num_tokens = 0



    def num_blocks_needed(self, num_new_tokens: int) -> int:
        """Number of new blocks that appending ``num_new_tokens`` would allocate.

        Does not change the table or the pool.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be >= 0, got {num_new_tokens}")
        return (self._num_tokens + num_new_tokens + self.block_size - 1) // self.block_size - len(self._blocks)


    def can_append(self, num_new_tokens: int) -> bool:
        """True if the pool has enough free blocks for ``num_new_tokens``.

        Does not change the table or the pool.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be >= 0, got {num_new_tokens}")
        return self.pool.num_free >= self.num_blocks_needed(num_new_tokens)


    def append_tokens(self, num_new_tokens: int) -> None:
        """Grow the table by ``num_new_tokens`` tokens, allocating blocks as needed.

        Raises NoFreeBlocks when the pool cannot cover the whole append. In that
        case nothing changes: no block is allocated and num_tokens is unchanged.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be >= 0, got {num_new_tokens}")
        needed_blocks = self.num_blocks_needed(num_new_tokens)
        if self.pool.num_free < needed_blocks:
            raise NoFreeBlocks
        for _ in range(needed_blocks):
            block_id = self.pool.allocate()
            self._blocks.append(block_id)

        self._num_tokens += num_new_tokens

    def free_all(self) -> None:
        """Return every block this table holds to the pool and reset to empty."""
        for block_id in self._blocks:
            self.pool.free(block_id)
        self._blocks.clear()
        self._num_tokens = 0

    def adopt(self, block_ids: list[int], num_tokens: int) -> None:
        """Take blocks that someone else already allocated (the prefix cache), holding num_tokens tokens.

        The pool is not touched: the owner releases the blocks. The table must be empty, and
        block_ids must be exactly the blocks num_tokens tokens need. Raises ValueError otherwise.
        """
        if self._blocks or self._num_tokens:
            raise ValueError("adopt needs an empty table")
        if num_tokens < 0 or len(block_ids) != (num_tokens + self.block_size - 1) // self.block_size:
            raise ValueError(f"{len(block_ids)} blocks cannot hold {num_tokens} tokens")
        self._blocks = list(block_ids)
        self._num_tokens = num_tokens

    def forget(self) -> None:
        """Drop every block reference without returning anything to the pool.

        For a table whose blocks are owned by someone else (the prefix cache releases them).
        """
        self._blocks.clear()
        self._num_tokens = 0


    @property
    def block_ids(self) -> list[int]:
        """Block ids this table holds, in token order."""
        return list(self._blocks)

    @property
    def num_tokens(self) -> int:
        """Number of tokens currently stored in this table."""
        return self._num_tokens
