import random

import pytest

from minivllm.core.block_pool import BlockPool, NoFreeBlocks


def test_allocate_all_then_one_more_raises():
    pool = BlockPool(3)
    ids = [pool.allocate() for _ in range(3)]
    assert len(set(ids)) == 3
    with pytest.raises(NoFreeBlocks):
        pool.allocate()


def test_free_then_reallocate():
    pool = BlockPool(2)
    a = pool.allocate()
    pool.allocate()
    with pytest.raises(NoFreeBlocks):
        pool.allocate()
    pool.free(a)
    assert pool.allocate() == a


def test_double_free_raises():
    pool = BlockPool(4)
    b = pool.allocate()
    pool.free(b)
    with pytest.raises(ValueError):
        pool.free(b)


def test_free_out_of_range_raises():
    pool = BlockPool(4)
    with pytest.raises(ValueError):
        pool.free(4)
    with pytest.raises(ValueError):
        pool.free(-1)


def test_num_free_tracks_random_ops():
    # seeded so failures are reproducible
    rng = random.Random(1234)
    num_blocks = 32
    pool = BlockPool(num_blocks)
    held: set[int] = set()

    for _ in range(10_000):
        if held and (len(held) == num_blocks or rng.random() < 0.5):
            block = rng.choice(sorted(held))
            pool.free(block)
            held.remove(block)
        else:
            block = pool.allocate()
            assert block not in held
            held.add(block)
        assert pool.num_free == num_blocks - len(held)


# extra edge case
def test_fresh_pool_is_all_free():
    assert BlockPool(7).num_free == 7


# extra edge case
def test_zero_block_pool_allocate_raises():
    pool = BlockPool(0)
    assert pool.num_free == 0
    with pytest.raises(NoFreeBlocks):
        pool.allocate()


# extra edge case
def test_free_never_allocated_in_range_id_raises():
    pool = BlockPool(4)
    with pytest.raises(ValueError):
        pool.free(0)


# extra edge case
def test_negative_num_blocks_raises():
    with pytest.raises(ValueError):
        BlockPool(-1)


# extra edge case
def test_all_allocated_ids_are_unique_and_in_range():
    pool = BlockPool(16)
    ids = [pool.allocate() for _ in range(16)]
    assert sorted(ids) == list(range(16))
