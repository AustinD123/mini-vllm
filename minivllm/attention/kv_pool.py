"""KV pool: one preallocated tensor per cache, written by flat slot number.

Problem 19 — KV pool + scatter

Story
    The paged cache keeps keys and values for every block the engine owns. It
    is allocated once, up front, as a single tensor, and each token is written
    to the slot the block table gives it. Writes happen every step, for every
    running sequence, so they must be one vectorized scatter, not a Python loop
    per token.

Layout
    k and v are tensors of shape [num_layers, num_blocks, block_size, num_kv_heads, head_dim].
    A slot is a flat index: block = slot // block_size, offset = slot % block_size.

Operations
    KVPool(num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        allocates k and v once, filled with zeros
    write(layer, slots, k, v) -> None
        slots: 1-D tensor of N slot numbers
        k, v: [N, num_kv_heads, head_dim]
        stores row i of k and v at the slot slots[i], in the given layer

Errors
    Raises ValueError when any slot is outside 0 .. num_blocks * block_size - 1,
    or when k or v does not have shape [N, num_kv_heads, head_dim].

Example (num_layers=1, block_size=16, num_blocks=4)
    write(0, [37], k_token)      -> k[0, 2, 5] equals k_token   (slot 37 = block 2, offset 5)
    write(0, [14, 15, 16, 17], x) -> rows of x land at (block 0, 14), (0, 15), (1, 0), (1, 1)

Constraints
    write runs without a Python loop over tokens (one vectorized indexing operation).
    Memory is allocated once in KVPool(...); write never allocates a new full cache.

Edge cases to think about
    What is (block, offset) for slot 16? For slot 31?
    Why must a write that spans a block boundary still land in the right block?
    What should happen if the same slot appears twice in one write?

Tests
    tests/test_p19_kv_pool.py covers values readable at the expected (block, offset),
    a write that spans blocks, writes staying within their layer, a large write,
    and rejecting an out-of-range slot.
"""

from __future__ import annotations

import torch


class KVPool:
    """Preallocated K and V storage for every layer. See the module docstring."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Allocate k and v once, filled with zeros."""
        self.k = torch.zeros(num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.v = torch.zeros(num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store row i of ``k`` and ``v`` at flat slot ``slots[i]`` in ``layer``.

        Raises ValueError for an out-of-range slot or a shape mismatch.
        """
        if layer < 0 or layer >= self.k.shape[0]:
            raise ValueError(f"layer {layer} is out of range 0..{self.k.shape[0]-1}")
        if slots.ndim != 1:
            raise ValueError(f"slots must be 1-D, got {slots.shape}")
        if k.shape != (slots.shape[0], self.k.shape[3], self.k.shape[4]):
            raise ValueError(f"k must have shape [N, num_kv_heads, head_dim], got {k.shape}")
        if v.shape != (slots.shape[0], self.v.shape[3], self.v.shape[4]):
            raise ValueError(f"v must have shape [N, num_kv_heads, head_dim], got {v.shape}")
        num_blocks = self.k.shape[1]
        block_size = self.k.shape[2]
        if torch.any(slots < 0) or torch.any(slots >= num_blocks * block_size):
            raise ValueError(f"slots must be in range 0..{num_blocks * block_size - 1}")
        block_indices = slots // block_size
        offset_indices = slots % block_size
        self.k[layer, block_indices, offset_indices] = k
        self.v[layer, block_indices, offset_indices] = v

    
