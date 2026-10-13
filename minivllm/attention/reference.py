"""Reference causal attention, written by hand, checked against PyTorch.

Problem 17 — Hand-written causal attention

Story
    Attention is the one place where a token looks at every token before it.
    Before the paged-attention kernel, the engine needs a plain, readable version
    of the same computation to check against. If the fast path disagrees with
    this one, the bug is in the fast path. This file is the oracle.

Operations
    causal_attention(q, k, v) -> torch.Tensor
        q: [heads, Tq, dim]    k, v: [heads, Tk, dim]    Tq <= Tk
        returns [heads, Tq, dim]

Alignment
    The Tq queries are the last Tq positions of a Tk-long sequence. Query j sits
    at absolute position Tk - Tq + j, so it may attend to keys 0 .. Tk - Tq + j
    and nothing after that. When Tq == Tk this is ordinary causal attention.

Scores
    Each score is the dot product of a query and a key, scaled by 1 / sqrt(dim).
    Masked-out keys contribute nothing to the softmax.

Errors
    Raises ValueError when Tq > Tk, or when k and v do not match q's heads and dim.

Example
    heads = 1, Tq = Tk = 1: the only query may attend only to the only key,
        so the output equals v.
    heads = 1, Tq = 1, Tk = 3: the query sits at position 2 and sees all three keys.

Constraints
    Runs in O(heads * Tq * Tk * dim) time.
    Uses only tensor operations (no loops over the sequence).

Edge cases to think about
    Where does the mask go relative to the scaling and the softmax?
    With Tq < Tk, which keys does the first query see, and why?
    Why must masked scores become -inf rather than 0?

Tests
    tests/test_p17_causal_attention.py covers agreement with
    F.scaled_dot_product_attention(is_causal=True) within 1e-5, T = 1, and the
    non-square case where the queries are the last Tq of Tk keys.
"""

from __future__ import annotations

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention with the queries aligned to the end of the keys.

    Raises ValueError when Tq > Tk, or when k and v do not match q's heads and dim.
    """
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Heads dimension mismatch")
    if q.shape[2] != k.shape[2] or q.shape[2]!= v.shape[2]:
        raise ValueError("Dim dimension mismatch")
    Tq = q.shape[1]
    Tk = k.shape[1]
    if Tq > Tk:
        raise ValueError("Tq cannot be greater than Tk")
    # Compute the attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[2] ** 0.5)
    # Create a causal mask
    mask = torch.tril(torch.ones(Tq, Tk, device=q.device), diagonal=Tk - Tq).unsqueeze(0)  # [1, Tq, Tk]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    # Compute the output
    output = torch.matmul(attn_weights, v)
    return output
