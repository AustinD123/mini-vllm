"""Batched decode attention: one gather and one masked attention call for every decode sequence.

Problem 23b — Batched decode attention (spec written and implemented by the agent this round)

Story
    A decode step has one query per running sequence. paged_attention_single handles one sequence
    per call, so a batch of 32 costs 32 gathers and 32 attention calls in every layer. This module
    does the whole batch in one gather over a padded block table and one masked attention call.

Operations
    paged_decode_attention(q, pool, layer, block_tables, context_lens) -> torch.Tensor
        q             [B, Hq, D]         one query per sequence: its last position
        pool          KVPool; pool.k and pool.v are [layers, num_blocks, block_size, Hkv, D]
        layer         which layer of the pool to read
        block_tables  [B, max_blocks]    long, block ids in token order, padded on the right
                                         with any valid block id
        context_lens  [B]                long, tokens each sequence holds, including its query (>= 1)
        returns       [B, Hq, D]         same dtype as q

Shapes inside the function
    C              max_blocks * block_size   padded context length
    gathered K, V  [B, Hkv, C, D]            one gather for the whole batch
    scores         [B, Hkv, G, C]            G = Hq / Hkv query heads per kv head

Choices
    GQA: query head h uses kv head h // G. The query is reshaped to [B, Hkv, G, D], so each kv
    head is read once and never repeated (no repeat_interleave).
    Padding is NaN-safe. Positions at or beyond context_len get score -inf (softmax weight 0) and
    their V is set to 0 before the matmul. Setting only the weights to 0 is not enough, because
    0 * NaN is NaN and an unused block may hold NaN. A reserved null block would also work
    (vLLM's choice), but it needs a block id that the pool never hands out, and BlockPool
    accounting depends on every id being usable. Masking V keeps the pool unchanged.
    Softmax runs in fp32 (float64 stays float64). The probabilities are cast to V's dtype for
    the matmul, which accumulates in fp32 on the GPU.
    No Python loop over sequences: the number of kernels does not depend on B.

Errors
    Raises ValueError for a shape or head-count mismatch. It does not check context_lens against
    the table on the device (that would sync). The caller guarantees 1 <= context_len <= C.

Constraints
    Writes to the pool happen before this call (the backend writes this step's K and V first), so
    a decode query sees its own token.

Edge cases to think about
    Which keys does the query at position context_len - 1 see?
    What happens to a NaN in a padded block if only the scores are masked?
    Why must the mask cover V as well as the scores?

Tests
    tests/test_p23b_batched.py checks agreement with paged_attention_single for context lengths
    1, 15, 16, 17, 100, 300 in one batch, shuffled and non-contiguous block tables, a batch of one,
    the Qwen2.5-0.5B head layout, NaN in every unreferenced slot, and a kernel count that does not
    grow with B on CUDA.
"""

from __future__ import annotations

import torch

from minivllm.attention.kv_pool import KVPool


def paged_decode_attention(
    q: torch.Tensor,
    pool: KVPool,
    layer: int,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """Attend one query per sequence over its paged K and V, in one batched call. See the module docstring.

    Raises ValueError on a shape mismatch, on Hq not a multiple of Hkv, or when head_dim differs from the pool.
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape [B, Hq, D], got {tuple(q.shape)}")
    B, Hq, D = q.shape
    Hkv = pool.k.shape[3]
    block_size = pool.k.shape[2]
    if D != pool.k.shape[4]:
        raise ValueError(f"q has head_dim {D} but the pool stores {pool.k.shape[4]}")
    if Hkv == 0 or Hq % Hkv != 0:
        raise ValueError(f"q has {Hq} heads, which is not a multiple of the {Hkv} kv heads in the pool")
    if block_tables.ndim != 2 or block_tables.shape[0] != B:
        raise ValueError(f"block_tables must have shape [B, max_blocks] with B={B}, got {tuple(block_tables.shape)}")
    if context_lens.shape != (B,):
        raise ValueError(f"context_lens must have shape [B]={B}, got {tuple(context_lens.shape)}")
    device = pool.k.device
    tables = block_tables.to(device)
    lengths = context_lens.to(device)
    G = Hq // Hkv
    C = block_tables.shape[1] * block_size

    # One gather for the whole batch: [B, max_blocks, block_size, Hkv, D] -> [B, Hkv, C, D].
    k = pool.k[layer][tables].reshape(B, C, Hkv, D).permute(0, 2, 1, 3)
    v = pool.v[layer][tables].reshape(B, C, Hkv, D).permute(0, 2, 1, 3)
    valid = torch.arange(C, device=device)[None, :] < lengths[:, None]  # [B, C]
    # Zero V where the position is not used, so a NaN in an unused slot cannot reach the output.
    v = v.masked_fill(~valid[:, None, :, None], 0)

    accumulate = torch.promote_types(q.dtype, torch.float32)
    qg = q.reshape(B, Hkv, G, D)  # query head h = kv_head * G + g
    scores = torch.matmul(qg, k.transpose(-1, -2)).to(accumulate) * (D ** -0.5)  # [B, Hkv, G, C]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v)  # [B, Hkv, G, D]
    return out.reshape(B, Hq, D).to(q.dtype)
