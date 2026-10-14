"""Grouped-query causal attention: more query heads than key/value heads.

Problem 22 — GQA

Story
    Modern models such as Qwen2.5 use fewer key/value heads than query heads.
    Several query heads share one KV head, so the cache stores only the KV heads.
    The engine's attention has to accept queries with more heads than the K and V
    it reads, and still produce the same answer as ordinary causal attention would
    if each query head had its own copy of its KV head.

Dimensions
    Hq   = num_q_heads
    Hkv  = num_kv_heads, with Hq % Hkv == 0
    D    = head_dim
    Tq   = number of queries (the last Tq positions)
    Tk   = number of keys (Tq <= Tk)

Operations
    grouped_query_attention(q, k, v) -> torch.Tensor
        q: [Hq, Tq, D]     k, v: [Hkv, Tk, D]
        returns [Hq, Tq, D]

Grouping
    Query heads are split into consecutive groups of size Hq // Hkv. Every head in
    a group attends with the same K and V head. The result must equal what you get
    by repeating each KV head Hq // Hkv times along the head axis (repeat_interleave)
    and then running ordinary causal attention with the queries aligned to the end
    of the keys, as in problem 17.

Errors
    Raises ValueError when Hq is not a multiple of Hkv, when k and v do not have the
    same shape, when their head_dim differs from q's, or when Tq > Tk.

Example
    Hq = 4, Hkv = 2, Tq = Tk = 1, D = 3.
        q        [4, 1, 3]   any values
        k, v     [2, 1, 3]   v[0] and v[1] are different vectors
        output   [4, 1, 3]   one key per head, so the softmax weight is 1 and
                             output[h] = v[h // 2]:
                             output[0] = output[1] = v[0]
                             output[2] = output[3] = v[1]

Constraints
    Same complexity as problem 17: O(Hq * Tq * Tk * D) time.
    No Python loop over heads or over the sequence.

Edge cases to think about
    What is the group size when Hq == Hkv? When Hkv == 1 (multi-query attention)?
    Which KV head does query head 5 use when Hq = 14 and Hkv = 2?
    Does the causal alignment from problem 17 change when the heads are grouped?
    Should the function copy K and V once per query head, or can it read each KV
    head once for its whole group? What does each choice cost in memory?

Tests
    tests/test_p22_gqa.py checks agreement with a repeat-interleaved reference
    built from F.scaled_dot_product_attention, a decode-style case with Tq < Tk,
    the Hq == Hkv case, and the error cases.
"""

from __future__ import annotations

import torch


def grouped_query_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention where several query heads share one key/value head.

    See the module docstring. Raises ValueError on the shape errors listed there.
    """
    Hq = q.shape[0]
    Hkv = k.shape[0]
    if Hkv == 0:
        raise ValueError("k has 0 kv heads, so no query head has a kv head to use")
    if Hq == 0:
        raise ValueError("q has 0 heads, so there is nothing to attend with")
    if Hq % Hkv != 0:
        raise ValueError(f"q has {Hq} heads, which is not a multiple of the {Hkv} kv heads in k")
    if k.shape != v.shape:
        raise ValueError(f"k has shape {tuple(k.shape)} but v has shape {tuple(v.shape)}")
    D = q.shape[2]
    if D != k.shape[2]:
        raise ValueError(f"q has head_dim {D} but k has head_dim {k.shape[2]}")
    Tq = q.shape[1]
    Tk = k.shape[1]
    if Tq > Tk:
        raise ValueError(f"Tq {Tq} cannot be greater than Tk {Tk}")
    group = Hq // Hkv
    # Split the query heads into Hkv groups of `group` heads each. Each group shares one kv head,
    # so K and V are used as they are, without copies.
    q_grouped = q.reshape(Hkv, group, Tq, D)
    # Run ordinary causal attention with the queries aligned to the end of the keys.
    scores = torch.einsum("hgqd,hkd->hgqk", q_grouped, k) / (D ** 0.5)
    mask = torch.tril(torch.ones(Tq, Tk, device=q.device), diagonal=Tk - Tq) == 0
    scores = scores.masked_fill(mask, float("-inf"))
    attn_probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("hgqk,hkd->hgqd", attn_probs, v)
    return output.reshape(Hq, Tq, D)

