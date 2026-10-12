import pytest

from minivllm.core.block_pool import BlockPool, NoFreeBlocks
from minivllm.core.prefix_cache import PrefixCache

BLOCK = 16


def prompt(value: int) -> list[int]:
    return [value] * 17  # one full block plus one token


def test_freed_prefix_is_a_hit():
    cache = PrefixCache(BlockPool(4), BLOCK)
    cache.admit(0, prompt(1))
    cache.free(0)
    assert cache.admit(1, prompt(1)) == 16


def test_oldest_cached_block_is_evicted_first():
    pool = BlockPool(3)
    cache = PrefixCache(pool, BLOCK)
    cache.admit(0, prompt(1))  # A
    cache.free(0)
    cache.admit(1, prompt(2))  # B
    cache.free(1)
    cache.admit(2, prompt(3))  # C: pool is short, so A is evicted
    cache.free(2)
    assert cache.admit(3, prompt(2)) == 16  # B survived


def test_evicted_hash_is_no_longer_a_hit():
    pool = BlockPool(3)
    cache = PrefixCache(pool, BLOCK)
    cache.admit(0, prompt(1))
    cache.free(0)
    assert cache.admit(1, prompt(1)) == 16  # cached before any pressure
    cache.free(1)
    cache.admit(2, prompt(2))
    cache.free(2)
    cache.admit(3, prompt(3))  # pressure: A's block is evicted
    cache.free(3)
    assert cache.admit(4, prompt(1)) == 0


# extra edge case
def test_live_block_is_never_evicted():
    pool = BlockPool(3)
    cache = PrefixCache(pool, BLOCK)
    cache.admit(0, prompt(1))  # still in use, not cached
    with pytest.raises(NoFreeBlocks):
        cache.admit(1, prompt(2))  # needs 2 blocks, only 1 free, nothing cached
