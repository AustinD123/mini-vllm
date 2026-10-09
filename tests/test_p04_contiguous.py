import pytest

from minivllm.core.contiguous import ContiguousAllocator, NoSlotsAvailable, utilization

TOTAL = 4096
MAX_LEN = 1024


def test_only_four_sequences_admit_regardless_of_length():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    for seq_id in range(4):
        assert alloc.can_admit() is True
        alloc.admit(seq_id)
        alloc.append(seq_id, 3)  # short sequences still reserve max_len
    assert alloc.can_admit() is False
    with pytest.raises(NoSlotsAvailable):
        alloc.admit(4)


def test_utilization_matches_hand_computed_values():
    # lengths 10, 20, 30; block size 16
    # contiguous: 3 reservations of 1024 = 3072 slots reserved, 60 used
    assert utilization(60, 3 * MAX_LEN) == pytest.approx(60 / 3072)
    assert utilization(60, 3 * MAX_LEN) == pytest.approx(0.01953125)
    # paged: ceil(10/16)=1, ceil(20/16)=2, ceil(30/16)=2 blocks -> 5*16 = 80 slots
    assert utilization(60, 80) == pytest.approx(0.75)


def test_release_frees_a_slot_for_admission():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    for seq_id in range(4):
        alloc.admit(seq_id)
    alloc.release(2)
    assert alloc.can_admit() is True
    alloc.admit(4)


# extra edge case
def test_append_up_to_max_len_is_allowed():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    alloc.admit(0)
    alloc.append(0, MAX_LEN)


# extra edge case
def test_append_past_max_len_raises_and_changes_nothing():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    alloc.admit(0)
    alloc.append(0, MAX_LEN - 5)
    with pytest.raises(ValueError):
        alloc.append(0, 6)
    alloc.append(0, 5)  # the failed append left room exactly as it was


# extra edge case
def test_duplicate_admit_raises_value_error():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    alloc.admit(0)
    with pytest.raises(ValueError):
        alloc.admit(0)


# extra edge case
def test_append_and_release_unknown_sequence_raise_key_error():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    with pytest.raises(KeyError):
        alloc.append(9, 1)
    with pytest.raises(KeyError):
        alloc.release(9)


# extra edge case
def test_negative_append_raises_value_error():
    alloc = ContiguousAllocator(TOTAL, MAX_LEN)
    alloc.admit(0)
    with pytest.raises(ValueError):
        alloc.append(0, -1)


# extra edge case
def test_total_smaller_than_max_len_admits_nothing():
    alloc = ContiguousAllocator(512, MAX_LEN)
    assert alloc.can_admit() is False
    with pytest.raises(NoSlotsAvailable):
        alloc.admit(0)


# extra edge case
def test_utilization_zero_reserved_returns_zero():
    assert utilization(0, 0) == 0.0


# extra edge case
def test_utilization_rejects_used_greater_than_reserved():
    with pytest.raises(ValueError):
        utilization(10, 5)


# review: duplicate admit on a full allocator should raise ValueError (per docstring)
def test_duplicate_admit_when_full_raises_value_error():
    alloc = ContiguousAllocator(MAX_LEN, MAX_LEN)
    alloc.admit(0)
    with pytest.raises(ValueError):
        alloc.admit(0)


# review: spec constraint max_len >= 1 is not enforced
def test_max_len_zero_is_rejected():
    with pytest.raises(ValueError):
        ContiguousAllocator(TOTAL, 0)
