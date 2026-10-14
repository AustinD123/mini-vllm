import pytest

from minivllm.core.block_pool import BlockPool, NoFreeBlocks
from minivllm.core.prefix_cache import PrefixCache

BLOCK = 16
SHARED = list(range(32))


def test_shared_prefix_uses_fewer_blocks_than_independent_sequences():
    pool = BlockPool(10)
    cache = PrefixCache(pool, BLOCK)
    cache.admit(0, SHARED + [100] * 5)
    cache.mark_computed(0, 37)  # blocks are shareable once their prefill has run
    cache.admit(1, SHARED + [200] * 5)
    used = 10 - pool.num_free
    assert used == 4  # independent prompts would take 3 + 3 = 6


def test_freeing_one_sequence_leaves_shared_blocks_alive():
    pool = BlockPool(10)
    cache = PrefixCache(pool, BLOCK)
    cache.admit(0, SHARED + [100] * 5)
    cache.mark_computed(0, 37)  # prefill ran
    cache.admit(1, SHARED + [200] * 5)
    cache.mark_computed(1, 37)
    cache.free(0)
    assert 10 - pool.num_free == 3
    assert cache.admit(2, SHARED + [300] * 5) == 32


def test_fully_cached_prompt_recomputes_last_token():
    pool = BlockPool(10)
    cache = PrefixCache(pool, BLOCK)
    assert cache.admit(0, SHARED) == 0
    cache.mark_computed(0, 32)
    assert cache.admit(1, SHARED) == 16


# extra edge case
def test_fresh_prompt_has_no_cached_tokens():
    cache = PrefixCache(BlockPool(10), BLOCK)
    assert cache.admit(0, list(range(20))) == 0


# extra edge case
def test_freeing_everything_returns_all_blocks():
    pool = BlockPool(10)
    cache = PrefixCache(pool, BLOCK)
    cache.admit(0, SHARED + [100] * 5)
    cache.admit(1, SHARED + [200] * 5)
    cache.free(0)
    cache.free(1)
    assert pool.num_free + len(cache.lru) == 10


# extra edge case
def test_free_unknown_sequence_raises_key_error():
    cache = PrefixCache(BlockPool(10), BLOCK)
    with pytest.raises(KeyError):
        cache.free(42)


# extra edge case
def test_admit_when_pool_is_too_small_is_all_or_nothing():
    pool = BlockPool(1)
    cache = PrefixCache(pool, BLOCK)
    with pytest.raises(NoFreeBlocks):
        cache.admit(0, list(range(20)))  # needs 2 blocks
    assert pool.num_free == 1
