"""Paged attention for one sequence: gather its K and V through its block table, then attend.

Problem 20 — Paged attention, one sequence

Story
    A sequence's keys and values are not stored in one contiguous region. They sit
    in the blocks its block table names, scattered across the pool. To run attention,
    the engine has to gather them back into token order and attend over them. This
    is the first version that reads the real paged layout.

Dimensions
    H  = num_heads (the pool stores the same number of KV heads)
    D  = head_dim
    Tq = number of queries (the last Tq positions of the sequence)
    C  = context_len (tokens so far, C >= Tq)
    B  = block_size (pool.k.shape[2])

Operations
    paged_attention_single(q, pool, layer, block_table, context_len) -> torch.Tensor
        q            [H, Tq, D]          queries for the last Tq positions
        pool         KVPool; pool.k and pool.v are [layers, num_blocks, B, H, D]
        layer        which layer of the pool to read
        block_table  list of block ids, in token order, length >= ceil(C / B)
        context_len  C
        returns      [H, Tq, D]

Shapes inside the function
    slots           [C]          flat slot of each token 0 .. C-1
    pool.k[layer]   [num_blocks, B, H, D]
    gathered K      [H, C, D]    token order, after the gather
    gathered V      [H, C, D]
    scores          [H, Tq, C]   q @ K^T, scaled by 1 / sqrt(D)
    output          [H, Tq, D]   softmax(scores) @ V

Attention rule
    Query j sits at position context_len - Tq + j and attends to positions
    0 .. context_len - Tq + j, scaled by 1 / sqrt(head_dim). This is the same
    alignment as problem 17.

Errors
    Raises ValueError when Tq > context_len, or when context_len needs more
    blocks than block_table holds.

Example (block_size = 16, block_table = [2, 0], context_len = 21)
    token 0 is at slot 32 (block 2, offset 0); token 16 is at slot 0 (block 0, offset 0).
    The gather returns K and V in token order 0..20, even though block 2 comes first.
    With B = 16, C = 21: slots has shape [21]; the gathered K and V are [H, 21, D].

Constraints
    Uses the slot mapping from problem 2 to find each token's slot.
    Gathers all context K and V once, then attends (no per-token loop over attention).

Edge cases to think about
    Which slot is token 16 when block_table = [2, 0]?
    What changes when context_len is not a multiple of block_size?
    What happens to a block table that is longer than needed?

Tests
    tests/test_p20_paged_attention.py checks agreement with ordinary causal attention
    on the same K and V, a context length that is not a multiple of the block size,
    a shuffled block table, and a single-query decode step.
"""

from __future__ import annotations

import torch

from minivllm.attention.kv_pool import KVPool
from minivllm.core.block_table import slot_mapping


def paged_attention_single(
    q: torch.Tensor,
    pool: KVPool,
    layer: int,
    block_table: list[int],
    context_len: int,
) -> torch.Tensor:
    """Attend over one sequence's paged K and V. See the module docstring.

    Raises ValueError when Tq > context_len, or when block_table cannot cover context_len.
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape [num_heads, Tq, head_dim], got {q.shape}")
    if q.shape[0] != pool.k.shape[3]:
        raise ValueError(f"q has {q.shape[0]} heads but the pool stores {pool.k.shape[3]} kv heads")
    if q.shape[2] != pool.k.shape[4]:
        raise ValueError(f"q has head_dim {q.shape[2]} but the pool stores {pool.k.shape[4]}")
    Tq = q.shape[1]
    if Tq > context_len:
        raise ValueError(f"Tq {Tq} cannot be greater than context_len {context_len}")
    block_size = pool.k.shape[2]
    num_blocks_needed = (context_len + block_size - 1) // block_size
    if num_blocks_needed > len(block_table):
        raise ValueError(f"block_table of length {len(block_table)} cannot cover context_len {context_len} with block_size {block_size}")
    # Gather K and V in token order with one indexed lookup, then attend.
    slots = torch.tensor(slot_mapping(block_table, 0, context_len, block_size), dtype=torch.long)
    k = pool.k[layer].reshape(-1, pool.k.shape[3], pool.k.shape[4])[slots].transpose(0, 1)
    v = pool.v[layer].reshape(-1, pool.v.shape[3], pool.v.shape[4])[slots].transpose(0, 1)
    # Now k and v are [num_heads, context_len, head_dim]. Attend with the same rule as problem 17.
    head_dim = k.shape[2]
    attn_scores = torch.einsum("hqd,hkd->hqk", q, k) / (head_dim ** 0.5)
    # Mask out keys beyond the context length for each query. Query j sits at position context_len - Tq + j.
    mask = torch.arange(context_len).unsqueeze(0) > (context_len - Tq + torch.arange(Tq).unsqueeze(1))
    attn_scores = attn_scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.einsum("hqk,hkd->hqd", attn_probs, v)
    return output
