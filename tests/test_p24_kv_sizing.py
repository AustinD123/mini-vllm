from types import SimpleNamespace

import pytest
import torch

from minivllm.attention.kv_pool import KVPool
from minivllm.runner.sizing import kv_bytes_per_token, num_blocks_for_memory

# Qwen2.5-0.5B-shaped config. head_dim is given explicitly here (896 / 14 = 64).
QWEN = SimpleNamespace(
    num_hidden_layers=24,
    num_key_value_heads=2,
    num_attention_heads=14,
    hidden_size=896,
    head_dim=64,
)
BLOCK = 16


def test_kv_bytes_per_token_fp16():
    assert kv_bytes_per_token(QWEN, torch.float16) == 12288


def test_kv_bytes_per_token_fp32_is_double():
    assert kv_bytes_per_token(QWEN, torch.float32) == 24576


def test_num_blocks_exact_boundary():
    free = 10 * BLOCK * 12288  # 1966080 bytes: exactly ten blocks
    assert num_blocks_for_memory(free, QWEN, torch.float16, BLOCK) == 10


def test_num_blocks_one_byte_short_rounds_down():
    free = 10 * BLOCK * 12288 - 1
    assert num_blocks_for_memory(free, QWEN, torch.float16, BLOCK) == 9


def test_head_dim_falls_back_to_hidden_over_heads_when_missing():
    no_head_dim = SimpleNamespace(
        num_hidden_layers=24,
        num_key_value_heads=2,
        num_attention_heads=14,
        hidden_size=896,
    )
    assert kv_bytes_per_token(no_head_dim, torch.float16) == 12288


def test_head_dim_none_falls_back_to_hidden_over_heads():
    none_head_dim = SimpleNamespace(
        num_hidden_layers=24,
        num_key_value_heads=2,
        num_attention_heads=14,
        hidden_size=896,
        head_dim=None,
    )
    assert kv_bytes_per_token(none_head_dim, torch.float16) == 12288


# review: a pool sized for fp16 must actually fit in the fp16 budget it was sized from
def test_pool_built_from_fp16_sizing_fits_the_budget():
    num_blocks = 10
    free = num_blocks * BLOCK * kv_bytes_per_token(QWEN, torch.float16)
    assert num_blocks_for_memory(free, QWEN, torch.float16, BLOCK) == num_blocks
    pool = KVPool(QWEN.num_hidden_layers, num_blocks, BLOCK, QWEN.num_key_value_heads, QWEN.head_dim, dtype=torch.float16)
    pool_bytes = (pool.k.numel() + pool.v.numel()) * pool.k.element_size()
    assert pool_bytes <= free


# review: the block count is an int, even when free_bytes is passed as a float
def test_num_blocks_returns_int_for_float_budget():
    free = float(10 * BLOCK * 12288)
    assert isinstance(num_blocks_for_memory(free, QWEN, torch.float16, BLOCK), int)


# extra edge case
def test_zero_free_bytes_gives_zero_blocks():
    assert num_blocks_for_memory(0, QWEN, torch.float16, BLOCK) == 0


# extra edge case
def test_less_than_one_block_gives_zero_blocks():
    free = BLOCK * 12288 - 1
    assert num_blocks_for_memory(free, QWEN, torch.float16, BLOCK) == 0


# extra edge case
def test_negative_free_bytes_raises():
    with pytest.raises(ValueError):
        num_blocks_for_memory(-1, QWEN, torch.float16, BLOCK)


# extra edge case
def test_non_positive_block_size_raises():
    with pytest.raises(ValueError):
        num_blocks_for_memory(1024, QWEN, torch.float16, 0)
