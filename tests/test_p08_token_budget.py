import pytest

from minivllm.core.scheduler import split_budget
from minivllm.core.sequence import Sequence


def decode_seq(seq_id: int) -> Sequence:
    """A running sequence with exactly one uncomputed token (its last output)."""
    seq = Sequence(seq_id=seq_id, prompt_len=4, max_new_tokens=10, arrival_time=float(seq_id))
    seq.on_tokens_computed(4, sampled=True)
    return seq


def prefill_seq(seq_id: int, prompt_len: int) -> Sequence:
    return Sequence(seq_id=seq_id, prompt_len=prompt_len, max_new_tokens=5, arrival_time=float(seq_id))


def ten_decodes() -> list[Sequence]:
    return [decode_seq(i) for i in range(10)]


def test_budget_64_with_ten_decodes_leaves_54_for_prefill():
    out = split_budget(ten_decodes(), [prefill_seq(10, 54)], token_budget=64)
    assert out[10] == 54
    assert sum(out.values()) == 64


def test_budget_smaller_than_decodes_schedules_first_decodes_only():
    out = split_budget(ten_decodes(), [prefill_seq(10, 2)], token_budget=4)
    assert out == {0: 1, 1: 1, 2: 1, 3: 1}


# extra edge case
def test_prefill_that_does_not_fit_blocks_the_ones_behind_it():
    out = split_budget([], [prefill_seq(10, 55), prefill_seq(11, 10)], token_budget=54)
    assert 10 not in out
    assert 11 not in out


# extra edge case
def test_prefill_of_exactly_remaining_budget_fits():
    out = split_budget([], [prefill_seq(10, 54)], token_budget=54)
    assert out == {10: 54}


# extra edge case
def test_each_decode_gets_one_token():
    out = split_budget([decode_seq(0)], [], token_budget=64)
    assert out == {0: 1}


# extra edge case
def test_zero_budget_schedules_nothing():
    out = split_budget(ten_decodes(), [prefill_seq(10, 1)], token_budget=0)
    assert out == {}


# extra edge case
def test_no_work_returns_empty():
    assert split_budget([], [], token_budget=64) == {}


# extra edge case
def test_negative_budget_raises():
    with pytest.raises(ValueError):
        split_budget([], [], token_budget=-1)


# review: a prefill with nothing uncomputed should not be scheduled
def test_prefill_with_no_uncomputed_tokens_is_not_scheduled():
    done = prefill_seq(10, 4)
    done.on_tokens_computed(4, sampled=False)
    out = split_budget([], [done], token_budget=64)
    assert 10 not in out

