import pytest

from minivllm.core.prefix_cache import block_hashes

BLOCK = 16


def test_prompts_sharing_40_tokens_share_two_block_hashes():
    shared = list(range(40))
    a = block_hashes(shared + [900] * 16, BLOCK)
    b = block_hashes(shared + [901] * 16, BLOCK)
    assert len(a) == 3 and len(b) == 3
    assert a[:2] == b[:2]
    assert a[2] != b[2]


def test_identical_blocks_at_different_prefixes_hash_differently():
    x = block_hashes([7] * 16, BLOCK)
    y = block_hashes([1] * 16 + [7] * 16, BLOCK)
    assert y[1] != x[0]


def test_trailing_partial_block_gets_no_hash():
    assert len(block_hashes(list(range(15)), BLOCK)) == 0
    assert len(block_hashes(list(range(20)), BLOCK)) == 1
    assert len(block_hashes(list(range(32)), BLOCK)) == 2


# extra edge case
def test_same_tokens_give_same_hashes():
    tokens = list(range(50))
    assert block_hashes(tokens, BLOCK) == block_hashes(list(tokens), BLOCK)


# extra edge case
def test_empty_input_gives_no_hashes():
    assert block_hashes([], BLOCK) == []


# extra edge case
def test_block_size_below_one_raises():
    with pytest.raises(ValueError):
        block_hashes([1, 2, 3], 0)
