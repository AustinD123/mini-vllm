"""HF attention backend that writes K/V into the paged pool and attends through block tables.

Problem 23 — HF attention backend + exactness

Story
    Hugging Face calls one attention function per layer, and that function decides
    where K and V live. Until now the model has used its own contiguous cache. This
    backend replaces that call: it writes this step's new K and V into the shared
    pool at their slots, then attends each sequence over its own block table. The
    engine's runner sets up what each step contains, and the model's output must
    match Hugging Face's generate() token for token, whatever the batching.

Operations
    paged_backend_forward(module, query, key, value, attention_mask, dropout=0.0,
                          scaling=None, is_causal=None, **kwargs) -> (output, None)
        module       the attention module; module.layer_idx says which pool layer to use
        query        [1, Hq, T, D]     all packed tokens of this step, head-major
        key, value   [1, Hkv, T, D]    the new K and V for those tokens, not yet repeated
        kwargs["kv_context"]           a StepContext (see below); required
        output       [1, T, Hq, D]     the transposed layout Hugging Face expects

    register_backend() -> None
        makes paged_backend_forward available under BACKEND_NAME

    StepContext(pool, slots, seqs)
        pool    the KVPool shared by all sequences
        slots   [T] long tensor: the flat slot each packed token is written to
        seqs    list of SeqSlice, which tile the packed row 0 .. T-1 in order

    SeqSlice(start, length, block_table, context_len)
        start        index of this sequence's first token in the packed row
        length       number of tokens this sequence contributes this step
        block_table  the blocks holding this sequence, in token order
        context_len  tokens this sequence holds after this step (its last query is at
                     position context_len - 1)

Choices (the spec leaves these open)
    The attention_mask argument is ignored. Each sequence's causal rule comes from
    its SeqSlice, so the backend does not need Hugging Face's mask.
    The scaling argument is the 1 / sqrt(head_dim) that Hugging Face passes in; it
    is fine to use it or to compute the same value from head_dim.

Errors
    Raises ValueError when kv_context is missing, when the seqs do not tile the
    packed row, or when a sequence's block table cannot cover its context_len.

Constraints
    Query heads can outnumber kv heads (problem 22). The pool stores kv heads only.
    Writes go through KVPool.write, one call per layer. No Python loop over tokens.

Edge cases to think about
    Which keys does a decode query see, and which blocks hold them?
    What must happen when a chunk of a prefill ends in the middle of a block?
    Does the same pool layer get written and read in one step, and in what order?
    What is the output layout Hugging Face expects, and which dimension moves?

Tests
    tests/test_p23_backend.py checks prefill logits against Hugging Face, greedy
    output against generate() across a block boundary with a shuffled block table,
    chunked prefill, a shared prefix block, recompute after preemption, and a packed
    batch of two sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AttentionInterface

from minivllm.attention.gqa import grouped_query_attention
from minivllm.attention.kv_pool import KVPool
from minivllm.core.block_table import slot_mapping

BACKEND_NAME = "minivllm_paged"


@dataclass
class SeqSlice:
    """One sequence's part of a packed step. See the module docstring."""

    start: int
    length: int
    block_table: list[int]
    context_len: int


@dataclass
class StepContext:
    """Everything the backend needs for one packed forward pass. See the module docstring."""

    pool: KVPool
    slots: torch.Tensor
    seqs: list[SeqSlice]


def paged_backend_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Write this step's K and V to the pool, then attend per sequence. See the module docstring.

    Raises ValueError when kwargs has no kv_context, when the seqs do not tile the packed row,
    when a block table cannot cover its context_len, or when q, k, v and the pool are on different devices.
    Raises NotImplementedError for a sliding-window layer or a scaling other than 1 / sqrt(head_dim).
    The attention_mask argument is accepted in any shape and ignored.
    """
    if "kv_context" not in kwargs:
        raise ValueError("paged_backend_forward requires a StepContext in kwargs['kv_context']")
    if not isinstance(kwargs["kv_context"], StepContext):
        raise ValueError("kwargs['kv_context'] must be a StepContext")
    context: StepContext = kwargs["kv_context"]
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(f"query must have shape [1, Hq, T, D], got {query.shape}")
    if key.ndim != 4 or key.shape[0] != 1:
        raise ValueError(f"key must have shape [1, Hkv, T, D], got {key.shape}")
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"value must have shape [1, Hkv, T, D], got {value.shape}")    
    if context.pool is None:
        raise ValueError("StepContext.pool is None; paged_backend_forward requires a KVPool")
    if context.slots is None:
        raise ValueError("StepContext.slots is None; paged_backend_forward requires a slots tensor")
    if context.seqs is None:
        raise ValueError("StepContext.seqs is None; paged_backend_forward requires a list of SeqSlice")
    if not isinstance(context.seqs, list):
        raise ValueError("StepContext.seqs must be a list of SeqSlice") 
    for seq in context.seqs:
        if not isinstance(seq, SeqSlice):
            raise ValueError("StepContext.seqs must be a list of SeqSlice") 
    if not context.seqs:
        raise ValueError("StepContext.seqs is empty; paged_backend_forward requires at least one SeqSlice") 
    T = query.shape[2]
    expected_start = 0
    for seq in context.seqs:
        if seq.start != expected_start:
            raise ValueError(f"seqs must tile the packed row: expected start {expected_start}, got {seq.start}")
        expected_start += seq.length
    if expected_start != T:
        raise ValueError(f"seqs cover {expected_start} tokens but the packed row has {T}")
    if context.slots.shape[0] != T:
        raise ValueError(f"slots has {context.slots.shape[0]} entries but the packed row has {T} tokens")
    if getattr(module, "sliding_window", None) is not None:
        raise NotImplementedError("sliding-window attention is not supported by the paged backend")
    head_dim = query.shape[3]
    if scaling is not None and abs(scaling - head_dim ** -0.5) > 1e-12:
        raise NotImplementedError(f"scaling {scaling} differs from 1/sqrt(head_dim); the paged backend only supports the default")
    pool = context.pool
    if query.device != pool.k.device or key.device != pool.k.device or value.device != pool.k.device:
        raise ValueError(f"q, k, v and the pool must be on the same device; pool is on {pool.k.device}, q is on {query.device}")
    layer = module.layer_idx
    # Write first, so this step's own tokens are visible to their queries.
    k = key[0].transpose(0, 1)  # [T, Hkv, D]
    v = value[0].transpose(0, 1)  # [T, Hkv, D]
    block_size = pool.k.shape[2]
    pool.write(layer, context.slots.to(pool.k.device), k, v)
    outputs = []
    for seq in context.seqs:
        if seq.context_len > len(seq.block_table) * block_size:
            raise ValueError(f"block_table of length {len(seq.block_table)} cannot cover context_len {seq.context_len} with block_size {block_size}")
        q = query[0, :, seq.start:seq.start + seq.length]  # [Hq, L, D]
        slots = torch.tensor(slot_mapping(seq.block_table, 0, seq.context_len, block_size), dtype=torch.long, device=pool.k.device)
        k_ctx = pool.k[layer].reshape(-1, pool.k.shape[3], pool.k.shape[4])[slots].transpose(0, 1)  # [Hkv, C, D]
        v_ctx = pool.v[layer].reshape(-1, pool.v.shape[3], pool.v.shape[4])[slots].transpose(0, 1)  # [Hkv, C, D]
        outputs.append(grouped_query_attention(q, k_ctx, v_ctx))  # [Hq, L, D]
    out = torch.cat(outputs, dim=1)  # [Hq, T, D]
    return out.transpose(0, 1).unsqueeze(0), None  # [1, T, Hq, D]


def register_backend() -> None:
    """Make paged_backend_forward available to Hugging Face under BACKEND_NAME.

    Call once before building a model with attn_implementation=BACKEND_NAME.
    """
    AttentionInterface.register(BACKEND_NAME, paged_backend_forward)
