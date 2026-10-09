import pytest

from minivllm.core.block_pool import BlockPool, NoFreeBlocks
from minivllm.core.block_table import BlockTable

BLOCK_SIZE = 16


def test_one_token_at_a_time_allocates_at_0_16_32():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    allocs = []
    for i in range(33):
        before = pool.num_free
        bt.append_tokens(1)
        if pool.num_free < before:
            allocs.append(i)
    assert allocs == [0, 16, 32]
    assert bt.num_tokens == 33


def test_append_40_at_once_allocates_3():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    before = pool.num_free
    bt.append_tokens(40)
    assert before - pool.num_free == 3
    assert bt.num_tokens == 40


def test_can_append_false_when_pool_cannot_cover():
    pool = BlockPool(2)
    bt = BlockTable(pool, BLOCK_SIZE)
    assert bt.can_append(32) is True
    assert bt.can_append(33) is False


def test_failed_append_is_atomic():
    pool = BlockPool(2)
    bt = BlockTable(pool, BLOCK_SIZE)
    with pytest.raises(NoFreeBlocks):
        bt.append_tokens(33)
    assert pool.num_free == 2
    assert bt.num_tokens == 0
    bt.append_tokens(32)  # still works after the failed attempt
    assert bt.num_tokens == 32


def test_free_all_returns_every_block():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    bt.append_tokens(40)
    bt.free_all()
    assert pool.num_free == 10
    assert bt.num_tokens == 0


# extra edge case
def test_num_blocks_needed_fits_in_partial_block():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    bt.append_tokens(15)
    assert bt.num_blocks_needed(1) == 0
    assert bt.num_blocks_needed(2) == 1


# extra edge case
def test_num_blocks_needed_empty_table_zero_tokens():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    assert bt.num_blocks_needed(0) == 0


# extra edge case
def test_append_zero_tokens_changes_nothing():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    bt.append_tokens(0)
    assert pool.num_free == 10
    assert bt.num_tokens == 0


# extra edge case
def test_free_all_on_empty_table_is_noop():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    bt.free_all()
    assert pool.num_free == 10


# extra edge case
def test_exact_fill_then_no_room():
    pool = BlockPool(3)
    bt = BlockTable(pool, BLOCK_SIZE)
    bt.append_tokens(48)
    assert pool.num_free == 0
    assert bt.can_append(1) is False
    with pytest.raises(NoFreeBlocks):
        bt.append_tokens(1)


# extra edge case
def test_table_reusable_after_free_all():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    bt.append_tokens(16)
    bt.free_all()
    bt.append_tokens(16)
    assert bt.num_tokens == 16
    assert pool.num_free == 9


# extra edge case
def test_negative_counts_raise_value_error():
    pool = BlockPool(10)
    bt = BlockTable(pool, BLOCK_SIZE)
    with pytest.raises(ValueError):
        bt.append_tokens(-1)
    with pytest.raises(ValueError):
        bt.can_append(-1)
    with pytest.raises(ValueError):
        bt.num_blocks_needed(-1)


# review: two tables sharing one pool (engine case) must still be atomic
def test_shared_pool_append_is_atomic():
    pool = BlockPool(3)
    bt1 = BlockTable(pool, BLOCK_SIZE)
    bt2 = BlockTable(pool, BLOCK_SIZE)
    bt1.append_tokens(32)  # takes 2 of the 3 blocks
    assert bt2.can_append(33) is False
    with pytest.raises(NoFreeBlocks):
        bt2.append_tokens(33)
    assert bt2.num_tokens == 0
    assert pool.num_free == 1
