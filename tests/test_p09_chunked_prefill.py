import pytest

from minivllm.core.scheduler import plan_chunks
from minivllm.core.sequence import Sequence


def prefill_seq(seq_id: int, prompt_len: int, max_new: int = 5) -> Sequence:
    return Sequence(seq_id=seq_id, prompt_len=prompt_len, max_new_tokens=max_new, arrival_time=float(seq_id))


def decode_seq(seq_id: int, max_new: int = 100) -> Sequence:
    """A running sequence with exactly one uncomputed token (its last output)."""
    seq = Sequence(seq_id=seq_id, prompt_len=4, max_new_tokens=max_new, arrival_time=float(seq_id))
    seq.on_tokens_computed(4, sampled=True)
    return seq


def run(decoding: list[Sequence], prefilling: list[Sequence], budget: int, steps: int):
    """Drive plan_chunks for ``steps`` steps, applying each step's results."""
    history = []
    for _ in range(steps):
        alloc, samples = plan_chunks(decoding, prefilling, budget)
        history.append((alloc, samples))
        by_id = {seq.seq_id: seq for seq in decoding + prefilling}
        for seq_id, n in alloc.items():
            by_id[seq_id].on_tokens_computed(n, sampled=seq_id in samples)
    return history


def test_thousand_token_prompt_splits_into_four_steps():
    history = run([], [prefill_seq(0, 1000)], budget=256, steps=4)
    chunks = [alloc[0] for alloc, _ in history]
    assert chunks == [256, 256, 256, 232]


def test_sample_only_on_the_last_chunk():
    history = run([], [prefill_seq(0, 1000)], budget=256, steps=4)
    samples = [s for _, s in history]
    assert samples == [[], [], [], [0]]


def test_decode_is_never_starved_alongside_long_prompt():
    history = run([decode_seq(1)], [prefill_seq(0, 1000)], budget=256, steps=4)
    for alloc, samples in history:
        assert alloc[1] == 1
        assert 1 in samples


# extra edge case
def test_budget_divides_prompt_exactly():
    history = run([], [prefill_seq(0, 512)], budget=256, steps=2)
    assert [alloc[0] for alloc, _ in history] == [256, 256]
    assert [s for _, s in history] == [[], [0]]


# extra edge case
def test_prompt_smaller_than_budget_is_one_step():
    history = run([], [prefill_seq(0, 100)], budget=256, steps=1)
    assert history[0] == ({0: 100}, [0])


# extra edge case
def test_prompt_exactly_budget_samples_in_one_step():
    history = run([], [prefill_seq(0, 256)], budget=256, steps=1)
    assert history[0] == ({0: 256}, [0])


# extra edge case
def test_chunks_add_up_to_prompt_length():
    history = run([], [prefill_seq(0, 1000)], budget=256, steps=4)
    assert sum(alloc[0] for alloc, _ in history) == 1000


# extra edge case
def test_zero_budget_schedules_nothing():
    alloc, samples = plan_chunks([decode_seq(1)], [prefill_seq(0, 10)], token_budget=0)
    assert alloc == {}
    assert samples == []


# extra edge case
def test_negative_budget_raises():
    with pytest.raises(ValueError):
        plan_chunks([], [], token_budget=-1)


# review: a prefill with nothing uncomputed must not be scheduled or sampled
def test_fully_computed_prefill_is_not_sampled():
    done = prefill_seq(0, 4)
    done.on_tokens_computed(4, sampled=False)
    alloc, samples = plan_chunks([], [done], token_budget=256)
    assert 0 not in alloc
    assert 0 not in samples


# review: a decode-list sequence with more than one uncomputed token must not sample early
def test_decode_list_sequence_with_prompt_left_does_not_sample():
    seq = prefill_seq(0, 10)
    alloc, samples = plan_chunks([seq], [], token_budget=256)
    assert 0 not in samples


# review: update must record a partial chunk even when nothing was sampled
def test_update_records_partial_chunk_without_sample():
    from minivllm.core.block_pool import BlockPool
    from minivllm.core.scheduler import Scheduler
    from minivllm.core.scheduler_output import SchedulerOutput

    sched = Scheduler(BlockPool(10), 16, max_batch=2, token_budget=1000)
    seq = prefill_seq(0, 10)
    sched.add_request(seq)
    sched.step()
    sched.update(SchedulerOutput({0: 4}), {})
    assert seq.num_computed_tokens == 4


# review: the live scheduler must split a long prompt across steps using the budget
def test_scheduler_splits_long_prompt_by_budget():
    from minivllm.core.block_pool import BlockPool
    from minivllm.core.scheduler import Scheduler

    sched = Scheduler(BlockPool(100), 16, max_batch=1, token_budget=256)
    seq = prefill_seq(0, 1000)
    sched.add_request(seq)
    chunks = []
    for _ in range(4):
        out = sched.step()
        chunks.append(out.num_tokens[0])
        sampled = {0: 0} if seq.num_uncomputed() == out.num_tokens[0] else {}
        sched.update(out, sampled)
    assert chunks == [256, 256, 256, 232]
    assert seq.output_len == 1
