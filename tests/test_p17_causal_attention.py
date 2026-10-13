import math

import pytest
import torch
import torch.nn.functional as F

from minivllm.attention.reference import causal_attention


def rand_qkv(heads: int, tq: int, tk: int, dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(heads, tq, dim, generator=g)
    k = torch.randn(heads, tk, dim, generator=g)
    v = torch.randn(heads, tk, dim, generator=g)
    return q, k, v


def test_matches_sdpa_causal_square():
    q, k, v = rand_qkv(heads=4, tq=8, tk=8, dim=16)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(causal_attention(q, k, v), expected, atol=1e-5)


def test_single_token():
    q, k, v = rand_qkv(heads=2, tq=1, tk=1, dim=8)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(causal_attention(q, k, v), expected, atol=1e-5)


def test_non_square_queries_are_last_tq_of_tk_keys():
    tq, tk = 3, 7
    q, k, v = rand_qkv(heads=2, tq=tq, tk=tk, dim=8)
    mask = torch.ones(tq, tk, dtype=torch.bool).tril(diagonal=tk - tq)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(causal_attention(q, k, v), expected, atol=1e-5)


# extra edge case
def test_output_shape():
    q, k, v = rand_qkv(heads=3, tq=2, tk=5, dim=4)
    assert causal_attention(q, k, v).shape == (3, 2, 4)


# extra edge case
def test_future_keys_do_not_change_earlier_outputs():
    q, k, v = rand_qkv(heads=2, tq=6, tk=6, dim=8)
    before = causal_attention(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, -1] += 5.0
    v2[:, -1] -= 5.0
    after = causal_attention(q, k2, v2)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-5)


# extra edge case
def test_more_queries_than_keys_raises():
    q, k, v = rand_qkv(heads=1, tq=4, tk=2, dim=4)
    with pytest.raises(ValueError):
        causal_attention(q, k, v)
