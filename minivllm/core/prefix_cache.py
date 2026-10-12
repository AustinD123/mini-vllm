"""Chained block hashing: one hash per full block, identifying the whole prefix.

Problem 14 — Chained block hashing

Story
    Many requests share a prefix, such as the same system prompt. If the
    engine can tell that two requests' first blocks hold the same tokens, it
    can reuse their KV cache instead of computing it again. A block's contents
    alone are not enough: the same 16 tokens mean something different after
    one prefix than after another, and the cached keys and values depend on
    everything before them. So each block's hash also folds in the hash of the
    block before it.

Operations
    block_hashes(token_ids, block_size) -> list[int]
        one hash per full block, in order. A trailing partial block gets no hash.

Chain rule
    hash_i = hash((hash_{i-1}, tuple of the tokens in block i))
    Choice: the first block uses 0 as its parent hash. (Python's hash of None
    is not stable across versions, so it is avoided.)

Errors
    Raises ValueError when block_size < 1.

Example (block_size = 4)
    tokens [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        -> 2 hashes: one for [1..4], one for [5..8] chained to the first.
           [9, 10] is a partial block, so it gets no hash.
    tokens [1, 2, 3, 4, 5, 6, 7, 8]
        -> the same first 2 hashes as above.
    tokens [5, 6, 7, 8]
        -> 1 hash, and it differs from the second hash above, because the
           block has a different parent.

Constraints
    Runs in O(len(token_ids)) time.
    The same tokens always give the same hashes.

Edge cases to think about
    What is the hash count for 15 tokens with block size 16? For 32 tokens?
    Why does [5, 6, 7, 8] not hash the same as the second block of [1..8]?
    Is the empty list valid input?

Tests
    tests/test_p14_prefix_hashing.py covers two prompts sharing 40 tokens sharing
    exactly two block hashes, identical blocks at different prefixes hashing
    differently, and a trailing partial block getting no hash.
"""

from __future__ import annotations

import hashlib

from minivllm.core.block_pool import BlockPool, NoFreeBlocks


def _chain_hash(parent_hash: int, block: list[int]) -> int:
    """SHA-256 over the parent hash and the block's tokens, as a 256-bit int."""
    data = parent_hash.to_bytes(32, "big") + b"".join(t.to_bytes(8, "big", signed=True) for t in block)
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def block_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """Return one chained hash per full block of ``token_ids``.

    Raises ValueError when block_size < 1.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if not token_ids:
        return []
    hashes: list[int] = []
    for i in range(0, len(token_ids), block_size):
        block = token_ids[i:i + block_size]
        if len(block) < block_size:
            break
        parent_hash = 0 if i == 0 else hashes[-1]
        block_hash = _chain_hash(parent_hash, block)
        hashes.append(block_hash)
    return hashes


class PrefixCache:
    """Shares full prompt blocks between sequences by refcount.

    Problem 15 — Refcounted sharing

    Story
        Requests with the same system prompt would each store their own copy of
        its KV cache. Sharing the full blocks of a common prefix saves memory and
        skips recomputing it. A block can only go back to the pool when the last
        sequence using it has released it, so each block carries a count of its
        users.

    Operations
        PrefixCache(pool, block_size)
            pool: the shared BlockPool; block_size: tokens per block
        admit(seq_id, token_ids) -> int
            reserve blocks for a new sequence's prompt. Reuses every full block
            whose chained hash (block_hashes, problem 14) is already cached and
            allocates the rest. Returns the number of prompt tokens already
            cached, which is where the sequence's computation starts.
        free(seq_id) -> None
            release a sequence's blocks. A block returns to the pool only when no
            sequence uses it any more.

    Choices (the spec leaves these open)
        Only full blocks are shared. A partial block is always private.
        Only full blocks lying wholly inside the first len(token_ids) - 1 tokens
        are reused, so the block holding the last prompt token is always private
        and recomputed (its logits are needed for sampling).
        A cache hit increments the block's count; a new block starts at 1.
        admit is all-or-nothing: if the pool can't cover the new blocks,
        nothing changes and NoFreeBlocks is raised.

    Errors
        admit raises NoFreeBlocks when the pool cannot cover the new blocks.
        free raises KeyError for a sequence that was never admitted.

    Example (block_size = 16, pool = BlockPool(10))
        shared = 32 tokens
        admit(0, shared + 5 tokens)   -> 0 cached;  pool uses 3 blocks
        admit(1, shared + 5 others)   -> 32 cached; pool uses 4 blocks in total
        free(0)                       -> pool uses 3 blocks (shared ones stay alive)
        admit(2, shared + 5 more)     -> 32 cached
        admit(3, shared)              -> 16 cached (block 2 holds the last token, so it is recomputed)

    Constraints
        admit runs in O(prompt tokens) time. free runs in O(blocks held).
        Shared blocks are never copied.

    Edge cases to think about
        Why is copying a shared block unnecessary when only full blocks are shared?
        What should admit do when the prompt is shorter than one block?
        What happens to the refcount when one sequence holds the same hash twice?
        If two sequences admit the same new prefix at once, which one owns the block?

    Tests
        tests/test_p15_refcounting.py covers two sequences sharing a prefix using
        fewer blocks than two independent ones, freeing one leaving the shared
        blocks alive, and a fully cached prompt recomputing its last token.
    """

    def __init__(self, pool: BlockPool, block_size: int) -> None:
        """Create an empty cache over ``pool``."""
        self.pool = pool
        self.block_size = block_size
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}
        self.block_refs: dict[int, int] = {}
        self.seq_blocks: dict[int, list[int]] = {}
        self.lru: dict[int, None] = {}

    def _take_block(self) -> int:
        """A truly free block if there is one, otherwise the least recently used cached block."""
        if self.pool.num_free > 0:
            return self.pool.allocate()
        return self._evict_least_recent()

    def admit(self, seq_id: int, token_ids: list[int]) -> int:
        """Reserve blocks for ``token_ids`` and return the number of cached prompt tokens.

        Raises NoFreeBlocks when the pool cannot cover the new blocks.
        """
        if seq_id in self.seq_blocks:
            raise ValueError(f"Sequence {seq_id} is already admitted")
        hashes = block_hashes(token_ids, self.block_size)
        max_hits = max(0, (len(token_ids) - 1) // self.block_size)
        hits = 0
        while hits < min(len(hashes), max_hits) and hashes[hits] in self.hash_to_block:
            hits += 1
        total_blocks = -(-len(token_ids) // self.block_size)
        hit_blocks = [self.hash_to_block[hashes[i]] for i in range(hits)]
        claimable = self.pool.num_free + len(self.lru) - sum(1 for b in hit_blocks if b in self.lru)
        if claimable < total_blocks - hits:
            raise NoFreeBlocks(f"Sequence {seq_id} needs {total_blocks - hits} blocks")
        blocks: list[int] = []
        for block in hit_blocks:
            self.lru.pop(block, None)
            self.block_refs[block] = self.block_refs.get(block, 0) + 1
            blocks.append(block)
        for i in range(hits, total_blocks):
            block = self._take_block()
            self.block_refs[block] = 1
            blocks.append(block)
            if i < len(hashes) and hashes[i] not in self.hash_to_block:
                self.hash_to_block[hashes[i]] = block
                self.block_to_hash[block] = hashes[i]
        self.seq_blocks[seq_id] = blocks
        return hits * self.block_size

    def free(self, seq_id: int) -> None:
        """Release ``seq_id``'s blocks. Raises KeyError if it was never admitted."""
        blocks = self.seq_blocks.pop(seq_id)
        for block in blocks:
            self.block_refs[block] -= 1
            if self.block_refs[block] == 0:
                del self.block_refs[block]
                if block in self.block_to_hash:
                    self.lru[block] = None
                else:
                    self.pool.free(block)

    def _evict_least_recent(self) -> int:
        """Evict the least recently used cached block and return its id.

        Problem 16 — LRU cache of freed blocks

        Story
            A request that finishes releases its prompt blocks, but the next
            request with the same system prompt usually arrives soon after. If
            freed blocks were thrown away at once, every such request would
            recompute the prefix. Keeping freed blocks cached, and reusing them
            until the pool actually needs the space, turns those repeats into
            hits.

        Operations
            Called when the pool has no truly free block and admit needs one.
            Also used by free: a block whose refcount reaches 0 keeps its hash
            and contents and joins the LRU instead of returning to the pool.

        Rules (a choice where the spec is open)
            allocate prefers a truly free block; only when there is none does it
            evict. Eviction takes the least recently used cached block, removes
            that block's hash from the lookup table, and returns its id.
            A cache hit on a cached block removes it from the LRU, since it is
            in use again.
            Raises NoFreeBlocks when nothing is cached either.

        Example (block_size = 16, pool = BlockPool(3))
            admit A (17 tokens) -> 2 blocks; free A -> A's full block is cached
            admit B (17 tokens) -> pool has room, so no eviction; free B -> B cached
            admit C (17 tokens) -> pool is short, so A's block is evicted first
            admit B again       -> hit (B was not the oldest), 16 cached tokens
            admit A again       -> miss, 0 cached tokens (A was evicted)

        Constraints
            Each eviction and lookup runs in O(1) time.

        Edge cases to think about
            Which block is least recently used when two were freed in one step?
            Does a hit on a cached block change its place in the LRU?
            What happens to a block that is still in use when pressure arrives?

        Tests
            tests/test_p16_lru_cache.py covers a freed prefix being a hit, the
            oldest cached block being evicted first, and an evicted hash no
            longer being a hit.
        """
        block = next(iter(self.lru))
        del self.lru[block]
        del self.hash_to_block[self.block_to_hash.pop(block)]
        return block
