import pytest

from minivllm.core.block_table import slot_for_position, slot_mapping

TABLE = [5, 2, 9]
BLOCK_SIZE = 16


def test_slot_for_position_examples():
    assert slot_for_position(TABLE, 0, BLOCK_SIZE) == 80
    assert slot_for_position(TABLE, 15, BLOCK_SIZE) == 95
    assert slot_for_position(TABLE, 16, BLOCK_SIZE) == 32
    assert slot_for_position(TABLE, 37, BLOCK_SIZE) == 149
    assert slot_for_position(TABLE, 47, BLOCK_SIZE) == 159


def test_position_past_end_of_table_raises():
    with pytest.raises(IndexError):
        slot_for_position(TABLE, 48, BLOCK_SIZE)


def test_slot_mapping_crossing_two_boundaries():
    # positions 14..33 cross the block boundaries at 16 and 32
    expected = [80 + p for p in (14, 15)] + [32 + p - 16 for p in range(16, 32)]
    expected += [9 * 16 + p - 32 for p in (32, 33)]
    assert slot_mapping(TABLE, 14, 34, BLOCK_SIZE) == expected


def test_slot_mapping_example():
    assert slot_mapping(TABLE, 14, 18, BLOCK_SIZE) == [94, 95, 32, 33]


def test_slot_mapping_empty_range():
    assert slot_mapping(TABLE, 3, 3, BLOCK_SIZE) == []


def test_slot_mapping_past_end_of_table_raises():
    with pytest.raises(IndexError):
        slot_mapping(TABLE, 40, 49, BLOCK_SIZE)


def test_block_size_one_is_block_id_per_position():
    table = [5, 2, 9]
    assert [slot_for_position(table, p, 1) for p in range(3)] == [5, 2, 9]
    assert slot_mapping(table, 0, 3, 1) == [5, 2, 9]


# extra edge case
def test_negative_position_raises():
    with pytest.raises(IndexError):
        slot_for_position(TABLE, -1, BLOCK_SIZE)


# extra edge case
def test_slot_mapping_matches_slot_for_position():
    expected = [slot_for_position(TABLE, p, BLOCK_SIZE) for p in range(10, 40)]
    assert slot_mapping(TABLE, 10, 40, BLOCK_SIZE) == expected


# extra edge case
def test_last_slot_of_last_block():
    assert slot_for_position(TABLE, 47, BLOCK_SIZE) == 9 * 16 + 15
