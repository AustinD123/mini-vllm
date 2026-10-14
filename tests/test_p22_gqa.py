import pytest
import torch
import torch.nn.functional as F

from minivllm.attention.gqa import grouped_query_attention

DIM = 8


def expected(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Repeat each KV head across its query group, then causal SDPA with end-aligned mask.

    q: [Hq, Tq, D], k and v: [Hkv, Tk, D].
    """
    group = q.shape[0] // k.shape[0]
    k_rep = k.repeat_interleave(group, dim=0)
    v_rep = v.repeat_interleave(group, dim=0)
    tq, tk = q.shape[1], k.shape[1]
    mask = torch.ones(tq, tk, dtype=torch.bool).tril(diagonal=tk - tq)
    return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)


def test_matches_repeat_interleaved_reference():
    torch.manual_seed(0)
    q = torch.randn(6, 10, DIM)
    k = torch.randn(2, 10, DIM)
    v = torch.randn(2, 10, DIM)
    out = grouped_query_attention(q, k, v)
    assert out.shape == (6, 10, DIM)
    assert torch.allclose(out, expected(q, k, v), atol=1e-5)


def test_decode_style_fewer_queries_than_keys():
    torch.manual_seed(1)
    q = torch.randn(14, 3, DIM)
    k = torch.randn(2, 21, DIM)
    v = torch.randn(2, 21, DIM)
    out = grouped_query_attention(q, k, v)
    assert torch.allclose(out, expected(q, k, v), atol=1e-5)


def test_single_kv_head_multi_query():
    torch.manual_seed(2)
    q = torch.randn(4, 5, DIM)
    k = torch.randn(1, 5, DIM)
    v = torch.randn(1, 5, DIM)
    out = grouped_query_attention(q, k, v)
    assert torch.allclose(out, expected(q, k, v), atol=1e-5)


def test_equal_heads_is_plain_causal_attention():
    torch.manual_seed(3)
    q = torch.randn(3, 6, DIM)
    k = torch.randn(3, 6, DIM)
    v = torch.randn(3, 6, DIM)
    out = grouped_query_attention(q, k, v)
    assert torch.allclose(out, expected(q, k, v), atol=1e-5)


def test_single_key_output_is_the_group_value():
    q = torch.randn(4, 1, DIM)
    k = torch.randn(2, 1, DIM)
    v = torch.randn(2, 1, DIM)
    out = grouped_query_attention(q, k, v)
    assert torch.allclose(out[0, 0], v[0, 0], atol=1e-5)
    assert torch.allclose(out[1, 0], v[0, 0], atol=1e-5)
    assert torch.allclose(out[2, 0], v[1, 0], atol=1e-5)
    assert torch.allclose(out[3, 0], v[1, 0], atol=1e-5)


# extra edge case
def test_query_heads_not_multiple_of_kv_heads_raises():
    q = torch.randn(5, 3, DIM)
    k = torch.randn(2, 3, DIM)
    v = torch.randn(2, 3, DIM)
    with pytest.raises(ValueError):
        grouped_query_attention(q, k, v)


# extra edge case
def test_k_and_v_shape_mismatch_raises():
    q = torch.randn(4, 3, DIM)
    k = torch.randn(2, 3, DIM)
    v = torch.randn(2, 4, DIM)
    with pytest.raises(ValueError):
        grouped_query_attention(q, k, v)


# extra edge case
def test_head_dim_mismatch_raises():
    q = torch.randn(4, 3, DIM)
    k = torch.randn(2, 3, DIM + 1)
    v = torch.randn(2, 3, DIM + 1)
    with pytest.raises(ValueError):
        grouped_query_attention(q, k, v)


# review: zero kv heads must raise ValueError, not ZeroDivisionError
def test_zero_kv_heads_raises_value_error():
    q = torch.randn(4, 3, DIM)
    k = torch.randn(0, 3, DIM)
    v = torch.randn(0, 3, DIM)
    with pytest.raises(ValueError):
        grouped_query_attention(q, k, v)


# review: zero query heads must raise ValueError (decided in review)
def test_zero_query_heads_raises_value_error():
    q = torch.randn(0, 3, DIM)
    k = torch.randn(2, 3, DIM)
    v = torch.randn(2, 3, DIM)
    with pytest.raises(ValueError):
        grouped_query_attention(q, k, v)


# extra edge case
def test_query_longer_than_keys_raises():
    q = torch.randn(4, 5, DIM)
    k = torch.randn(2, 3, DIM)
    v = torch.randn(2, 3, DIM)
    with pytest.raises(ValueError):
        grouped_query_attention(q, k, v)
