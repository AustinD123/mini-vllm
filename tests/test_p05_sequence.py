import pytest

from minivllm.core.sequence import SeqStatus, Sequence


def make_seq(prompt_len: int = 4, max_new_tokens: int = 2) -> Sequence:
    return Sequence(
        seq_id=0,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        arrival_time=0.0,
    )


def test_illegal_transition_finished_to_running_raises():
    seq = make_seq()
    seq.set_status(SeqStatus.RUNNING)
    seq.set_status(SeqStatus.FINISHED)
    with pytest.raises(ValueError):
        seq.set_status(SeqStatus.RUNNING)


def test_num_uncomputed_after_preempt_is_prompt_plus_output():
    seq = make_seq(prompt_len=4, max_new_tokens=5)
    seq.set_status(SeqStatus.RUNNING)
    seq.on_tokens_computed(4, sampled=True)
    seq.on_tokens_computed(1, sampled=True)
    seq.preempt_recompute()
    assert seq.num_uncomputed() == seq.prompt_len + seq.output_len


def test_decode_step_computes_exactly_one_token():
    seq = make_seq(prompt_len=4, max_new_tokens=5)
    seq.on_tokens_computed(4, sampled=True)
    before = seq.num_computed_tokens
    assert seq.num_uncomputed() == 1
    seq.on_tokens_computed(1, sampled=True)
    assert seq.num_computed_tokens == before + 1


def test_finishing_at_max_new_tokens():
    seq = make_seq(prompt_len=4, max_new_tokens=3)
    seq.on_tokens_computed(4, sampled=True)
    seq.on_tokens_computed(1, sampled=True)
    assert seq.is_finished() is False
    seq.on_tokens_computed(1, sampled=True)
    assert seq.is_finished() is True


# extra edge case
def test_num_uncomputed_fresh_sequence_is_prompt_len():
    assert make_seq(prompt_len=7).num_uncomputed() == 7


# extra edge case
def test_is_prefill_until_prompt_computed():
    seq = make_seq(prompt_len=4)
    assert seq.is_prefill() is True
    seq.on_tokens_computed(3, sampled=False)
    assert seq.is_prefill() is True
    seq.on_tokens_computed(1, sampled=True)
    assert seq.is_prefill() is False


# extra edge case
def test_is_prefill_true_again_after_preempt():
    seq = make_seq(prompt_len=4)
    seq.set_status(SeqStatus.RUNNING)
    seq.on_tokens_computed(4, sampled=True)
    seq.preempt_recompute()
    assert seq.is_prefill() is True


# extra edge case
def test_computing_more_than_uncomputed_raises():
    seq = make_seq(prompt_len=4)
    with pytest.raises(ValueError):
        seq.on_tokens_computed(5, sampled=False)


# extra edge case
def test_negative_n_raises():
    seq = make_seq()
    with pytest.raises(ValueError):
        seq.on_tokens_computed(-1, sampled=False)


# extra edge case
def test_partial_chunk_cannot_sample():
    seq = make_seq(prompt_len=4)
    with pytest.raises(ValueError):
        seq.on_tokens_computed(2, sampled=True)


# extra edge case
def test_unsampled_step_leaves_output_len_alone():
    seq = make_seq(prompt_len=4)
    seq.on_tokens_computed(2, sampled=False)
    assert seq.output_len == 0


# extra edge case
def test_legal_lifecycle_transitions():
    seq = make_seq()
    seq.set_status(SeqStatus.RUNNING)
    seq.set_status(SeqStatus.PREEMPTED)
    seq.set_status(SeqStatus.RUNNING)
    seq.set_status(SeqStatus.FINISHED)
    assert seq.status == SeqStatus.FINISHED


# review: output_len must never exceed max_new_tokens
def test_sampling_past_max_new_tokens_raises():
    seq = make_seq(prompt_len=1, max_new_tokens=1)
    seq.on_tokens_computed(1, sampled=True)
    assert seq.is_finished() is True
    with pytest.raises(ValueError):
        seq.on_tokens_computed(1, sampled=True)


# review: a rejected on_tokens_computed call must not change status
def test_rejected_compute_leaves_status_unchanged():
    seq = make_seq(prompt_len=4)
    with pytest.raises(ValueError):
        seq.on_tokens_computed(99, sampled=False)
    assert seq.status == SeqStatus.WAITING


# review: a rejected preempt_recompute must not reset progress
def test_rejected_preempt_leaves_progress_unchanged():
    seq = make_seq(prompt_len=4, max_new_tokens=5)
    seq.set_status(SeqStatus.RUNNING)
    seq.on_tokens_computed(4, sampled=False)
    seq.set_status(SeqStatus.FINISHED)
    with pytest.raises(ValueError):
        seq.preempt_recompute()
    assert seq.num_computed_tokens == 4
