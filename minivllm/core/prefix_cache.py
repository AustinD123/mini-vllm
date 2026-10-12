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
