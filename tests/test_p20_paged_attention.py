import pytest
import torch
import torch.nn.functional as F

from minivllm.attention.kv_pool import KVPool
from minivllm.attention.paged import paged_attention_single
from minivllm.core.block_table import slot_mapping

BLOCK = 16
HEADS = 2
DIM = 8


def fill(pool: KVPool, block_table: list[int], k: torch.Tensor, v: torch.Tensor) -> None:
    """Write token-major K and V ([T, heads, dim]) into the pool through the block table."""
    slots = torch.tensor(slot_mapping(block_table, 0, k.shape[0], BLOCK))
    pool.write(0, slots, k, v)


def expected(q: torch.Tensor, k_hd: torch.Tensor, v_hd: torch.Tensor) -> torch.Tensor:
    """Reference from SDPA with the same end-aligned mask. k_hd, v_hd: [heads, T, dim]."""
    tq, tk = q.shape[1], k_hd.shape[1]
    mask = torch.ones(tq, tk, dtype=torch.bool).tril(diagonal=tk - tq)
    return F.scaled_dot_product_attention(q, k_hd, v_hd, attn_mask=mask)


def test_matches_contiguous_attention_on_same_kv():
    torch.manual_seed(0)
    t = 32
    pool = KVPool(1, 4, BLOCK, HEADS, DIM)
    k = torch.randn(t, HEADS, DIM)
    v = torch.randn(t, HEADS, DIM)
    table = [0, 1]
    fill(pool, table, k, v)
    q = torch.randn(HEADS, t, DIM)
    out = paged_attention_single(q, pool, 0, table, t)
    assert torch.allclose(out, expected(q, k.transpose(0, 1), v.transpose(0, 1)), atol=1e-5)


def test_context_not_a_multiple_of_block_size():
    torch.manual_seed(1)
    t = 21
    pool = KVPool(1, 4, BLOCK, HEADS, DIM)
    k = torch.randn(t, HEADS, DIM)
    v = torch.randn(t, HEADS, DIM)
    table = [0, 1]
    fill(pool, table, k, v)
    q = torch.randn(HEADS, t, DIM)
    out = paged_attention_single(q, pool, 0, table, t)
    assert torch.allclose(out, expected(q, k.transpose(0, 1), v.transpose(0, 1)), atol=1e-5)


def test_shuffled_block_table():
    torch.manual_seed(2)
    t = 21
    pool = KVPool(1, 4, BLOCK, HEADS, DIM)
    k = torch.randn(t, HEADS, DIM)
    v = torch.randn(t, HEADS, DIM)
    table = [2, 0]
    fill(pool, table, k, v)
    q = torch.randn(HEADS, t, DIM)
    out = paged_attention_single(q, pool, 0, table, t)
    assert torch.allclose(out, expected(q, k.transpose(0, 1), v.transpose(0, 1)), atol=1e-5)


# extra edge case
def test_single_query_decode_matches_last_row_of_prefill():
    torch.manual_seed(3)
    t = 21
    pool = KVPool(1, 4, BLOCK, HEADS, DIM)
    k = torch.randn(t, HEADS, DIM)
    v = torch.randn(t, HEADS, DIM)
    table = [2, 0]
    fill(pool, table, k, v)
    q = torch.randn(HEADS, t, DIM)
    full = paged_attention_single(q, pool, 0, table, t)
    last = paged_attention_single(q[:, -1:], pool, 0, table, t)
    assert torch.allclose(last, full[:, -1:], atol=1e-5)


# extra edge case
def test_query_longer_than_context_raises():
    pool = KVPool(1, 4, BLOCK, HEADS, DIM)
    q = torch.randn(HEADS, 5, DIM)
    with pytest.raises(ValueError):
        paged_attention_single(q, pool, 0, [0], 3)


# review: queries with a different head count than the pool must be rejected with ValueError
def test_head_count_mismatch_raises_value_error():
    pool = KVPool(1, 4, BLOCK, HEADS, DIM)
    q = torch.randn(HEADS * 2, 3, DIM)
    with pytest.raises(ValueError):
        paged_attention_single(q, pool, 0, [0], 3)
