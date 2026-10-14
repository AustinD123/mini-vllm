"""KV pool sizing: how many bytes one token costs, and how many blocks fit in free memory.

Problem 24 — KV pool sizing

Story
    The pool is allocated once, up front, so the engine has to choose its size before
    any request arrives. The right size comes from memory: after the weights are
    loaded, whatever is left is split into fixed-size blocks. That count is the
    engine's whole budget for keys and values, so getting the per-token byte cost
    wrong means either wasting memory or running out of it mid-request.

Dimensions
    L    = num_hidden_layers
    Hkv  = num_key_value_heads (the pool stores kv heads, not query heads)
    D    = head_dim
    B    = bytes per element for the dtype

Operations
    kv_bytes_per_token(config, dtype) -> int
        bytes of K and V that one token occupies across all layers
    num_blocks_for_memory(free_bytes, config, dtype, block_size) -> int
        how many whole blocks of block_size tokens fit in free_bytes

Choices (the spec leaves these open)
    head_dim is read from config.head_dim when the attribute is present and not None.
    Otherwise it is hidden_size // num_attention_heads.
    dtype is a torch.dtype. Its byte size is the element size of that dtype.
    num_blocks_for_memory rounds down. It returns 0 when free_bytes is smaller than
    one block.

Errors
    Raises ValueError when free_bytes is negative, or when block_size is not positive.

Example (illustrative config: 24 layers, 2 kv heads, head_dim 64, fp16)
    kv_bytes_per_token(config, torch.float16)            -> 12288
        (2 for K and V) * 24 layers * 2 heads * 64 dims * 2 bytes
    block of 16 tokens costs 16 * 12288 = 196608 bytes
    num_blocks_for_memory(1966080, config, fp16, 16)     -> 10   (exactly 10 blocks)
    num_blocks_for_memory(1966079, config, fp16, 16)     -> 9    (one byte short of 10)

Constraints
    Both functions are O(1). They read the config and do arithmetic only.

Edge cases to think about
    What if the config has no head_dim attribute? What if it has head_dim = None?
    Why is num_key_value_heads the right count here and num_attention_heads wrong?
    What is the byte size of fp32 compared with fp16, and what does that do to the block count?
    What should happen when free_bytes is exactly zero?

Tests
    tests/test_p24_kv_sizing.py checks the hand-computed Qwen2.5-0.5B value in fp16,
    the block count at the exact boundary, the head_dim fallback, and the error cases.
"""

from __future__ import annotations

import torch


def kv_bytes_per_token(config, dtype: torch.dtype) -> int:
    """Bytes of K and V that one token occupies across every layer. See the module docstring."""
    return 2 * config.num_hidden_layers * config.num_key_value_heads * (
        config.head_dim if getattr(config, "head_dim", None) is not None else config.hidden_size // config.num_attention_heads
    ) * torch.tensor([], dtype=dtype).element_size()
    


def num_blocks_for_memory(free_bytes: int, config, dtype: torch.dtype, block_size: int) -> int:
    """Whole blocks of block_size tokens that fit in free_bytes. See the module docstring.

    Raises ValueError when free_bytes is negative or block_size is not positive.
    """
    if free_bytes < 0:
        raise ValueError(f"free_bytes {free_bytes} cannot be negative")
    if block_size <= 0:
        raise ValueError(f"block_size {block_size} must be positive")
    bytes_per_token = kv_bytes_per_token(config, dtype)
    bytes_per_block = bytes_per_token * block_size
    return free_bytes // bytes_per_block
