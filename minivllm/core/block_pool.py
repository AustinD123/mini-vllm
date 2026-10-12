"""Fixed-size block pool: hands out and takes back integer block ids.

Problem 1 — Block pool

Story
    The KV cache is split into fixed-size blocks. Every running sequence
    needs blocks to store its keys and values, and blocks are returned when a
    sequence finishes. The pool is the bookkeeper that decides which block ids
    are currently free and which are in use.

Operations
    BlockPool(num_blocks)   create a pool with ids 0..num_blocks-1, all free
    allocate() -> int       take one free id out of the pool and return it
    free(block_id) -> None  put an id back into the pool
    num_free -> int         how many ids are currently free (property)

Errors
    allocate()   raises NoFreeBlocks when no id is free.
    free(id)     raises ValueError when id is outside 0..num_blocks-1.
    free(id)     raises ValueError when id is not currently allocated
                 (double-free, or freeing an id that was never handed out).

Example
    pool = BlockPool(3)
    pool.num_free            -> 3
    a = pool.allocate()      -> one of {0, 1, 2}; call it a
    pool.num_free            -> 2
    b = pool.allocate()      -> one of the remaining ids; b != a
    c = pool.allocate()      -> the last remaining id; c != a and c != b
    pool.num_free            -> 0
    pool.allocate()          -> raises NoFreeBlocks
    pool.free(a)
    pool.num_free            -> 1
    pool.allocate()          -> returns a (the only free id)
    pool.free(b)
    pool.free(b)             -> raises ValueError (double-free)
    pool.free(3)             -> raises ValueError (out of range)

Constraints
    allocate() and free() run in O(1) time, regardless of num_blocks.
    num_free runs in O(1) time.
    num_blocks >= 0. A pool of size 0 is valid and is always empty.
    Ids handed out by allocate() are always distinct while they are held.

Edge cases to think about
    What does a fresh pool report for num_free?
    What happens when you allocate from an empty pool? Does the pool change?
    After freeing an id, which calls are now legal?
    Can the same id be returned by allocate() twice without a free in between?
    What does num_free say after a long sequence of mixed allocate/free calls?

Tests
    tests/test_p01_block_pool.py covers the example above, double-free,
    out-of-range ids, and num_free over 10k random operations.
"""

from __future__ import annotations


class NoFreeBlocks(Exception):
    """Raised by BlockPool.allocate when no block is free."""

class Block:
    """A single block in the pool.

    The pool is a linked list of blocks. Each block has a next pointer to the
    next free block, or None if it is the last free block. The pool keeps a
    pointer to the head of the list.
    """

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.next: Block | None = None


class BlockPool:
    """Pool of ``num_blocks`` block ids, identified by 0..num_blocks-1.

    See the module docstring for the full problem statement.
    """

    def __init__(self, num_blocks: int) -> None:
        """Create a pool with all ``num_blocks`` blocks free."""
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._allocated: set[int] = set()

    def allocate(self) -> int:
        """Take one free block id out of the pool and return it.

        Raises NoFreeBlocks when no block is free.
        """
        if not self._free:
            raise NoFreeBlocks
        block_id = self._free.pop()
        self._allocated.add(block_id)
        return block_id

    def free(self, block_id: int) -> None:
        """Return ``block_id`` to the pool.

        Raises ValueError when ``block_id`` is out of range or is not currently
        allocated.
        """
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block id {block_id} is out of range")
        if block_id not in self._allocated:
            raise ValueError(f"block id {block_id} is not allocated")
        self._allocated.remove(block_id)
        self._free.append(block_id)

    @property
    def free_block_ids(self) -> set[int]:
        """Ids that are currently free."""
        return set(self._free)

    @property
    def num_free(self) -> int:
        """Number of block ids currently free."""
        return len(self._free)
