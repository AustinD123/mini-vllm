import time

import pytest
import torch

from minivllm.attention.kv_pool import KVPool

LAYERS, BLOCKS, BLOCK, HEADS, DIM = 2, 4, 16, 2, 8


def make_pool() -> KVPool:
    return KVPool(LAYERS, BLOCKS, BLOCK, HEADS, DIM)


def test_value_readable_at_expected_block_and_offset():
    pool = make_pool()
    k = torch.randn(1, HEADS, DIM)
    v = torch.randn(1, HEADS, DIM)
    pool.write(0, torch.tensor([37]), k, v)  # slot 37 = block 2, offset 5
    assert torch.equal(pool.k[0, 2, 5], k[0])
    assert torch.equal(pool.v[0, 2, 5], v[0])


def test_write_spanning_blocks_lands_in_the_right_blocks():
    pool = make_pool()
    x = torch.randn(4, HEADS, DIM)
    pool.write(0, torch.tensor([14, 15, 16, 17]), x, x)
    assert torch.equal(pool.k[0, 0, 14], x[0])
    assert torch.equal(pool.k[0, 0, 15], x[1])
    assert torch.equal(pool.k[0, 1, 0], x[2])
    assert torch.equal(pool.k[0, 1, 1], x[3])


# extra edge case
def test_write_stays_within_its_layer():
    pool = make_pool()
    x = torch.randn(1, HEADS, DIM)
    pool.write(1, torch.tensor([3]), x, x)
    assert torch.equal(pool.k[1, 0, 3], x[0])
    assert torch.count_nonzero(pool.k[0]) == 0


# extra edge case
def test_large_write_is_vectorized_and_fast():
    big = KVPool(1, 4096, BLOCK, HEADS, DIM)  # 65,536 slots, enough for 50,000
    n = 50_000
    slots = torch.arange(n)
    x = torch.randn(n, HEADS, DIM)
    start = time.perf_counter()
    big.write(0, slots, x, x)
    elapsed = time.perf_counter() - start
    assert torch.equal(big.k[0].reshape(-1, HEADS, DIM)[:n], x)
    assert elapsed < 2.0


# extra edge case
def test_out_of_range_slot_raises():
    pool = make_pool()
    x = torch.randn(1, HEADS, DIM)
    with pytest.raises(ValueError):
        pool.write(0, torch.tensor([BLOCKS * BLOCK]), x, x)
