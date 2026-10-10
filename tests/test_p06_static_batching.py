from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.core.sequence import SeqStatus, Sequence
import pytest
from minivllm.core.static_scheduler import StaticScheduler


def make(seq_id: int, prompt_len: int, max_new: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new,
        arrival_time=float(seq_id),
    )


def run_one_step(sched: StaticScheduler) -> tuple[SchedulerOutput, list[Sequence]]:
    out = sched.step()
    sampled = {seq_id: 0 for seq_id in out.num_tokens}
    finished = sched.update(out, sampled)
    return out, finished


def test_batch_runs_for_longest_output_length():
    sched = StaticScheduler(batch_size=2)
    sched.add_request(make(0, 4, 10))
    sched.add_request(make(1, 4, 200))
    steps = 0
    while True:
        out, _ = run_one_step(sched)
        if not out.num_tokens:
            break
        steps += 1
    assert steps == 200


def test_no_admission_while_batch_is_running():
    sched = StaticScheduler(batch_size=2)
    sched.add_request(make(0, 4, 5))
    run_one_step(sched)
    sched.add_request(make(1, 4, 1))
    out, _ = run_one_step(sched)
    assert 1 not in out.num_tokens


# extra edge case
def test_batch_size_limits_first_batch():
    sched = StaticScheduler(batch_size=2)
    for i in range(3):
        sched.add_request(make(i, 4, 1))
    out, _ = run_one_step(sched)
    assert set(out.num_tokens) == {0, 1}


# extra edge case
def test_next_batch_starts_after_previous_finishes():
    sched = StaticScheduler(batch_size=2)
    for i in range(3):
        sched.add_request(make(i, 4, 1))
    run_one_step(sched)
    out, _ = run_one_step(sched)
    assert set(out.num_tokens) == {2}


# extra edge case
def test_update_returns_finished_sequences():
    sched = StaticScheduler(batch_size=1)
    sched.add_request(make(0, 4, 1))
    _, finished = run_one_step(sched)
    assert [seq.seq_id for seq in finished] == [0]


# extra edge case
def test_prefill_step_computes_whole_prompt():
    sched = StaticScheduler(batch_size=1)
    sched.add_request(make(0, 7, 3))
    out, _ = run_one_step(sched)
    assert out.num_tokens == {0: 7}


# extra edge case
def test_decode_step_computes_one_token():
    sched = StaticScheduler(batch_size=1)
    sched.add_request(make(0, 4, 3))
    run_one_step(sched)
    out, _ = run_one_step(sched)
    assert out.num_tokens == {0: 1}


# extra edge case
def test_empty_scheduler_returns_no_work():
    sched = StaticScheduler(batch_size=2)
    assert sched.step().num_tokens == {}


# extra edge case
def test_batch_size_below_one_raises():
    import pytest

    with pytest.raises(ValueError):
        StaticScheduler(batch_size=0)


# review: update must reject a sampled id that isn't in the scheduler's output
def test_update_rejects_unknown_sampled_id():
    sched = StaticScheduler(batch_size=1)
    sched.add_request(make(0, 4, 3))
    out = sched.step()
    with pytest.raises(ValueError):
        sched.update(out, {0: 0, 99: 0})


# review: admitted sequences must be marked RUNNING (scheduler owns status)
def test_admitted_sequence_is_marked_running():
    sched = StaticScheduler(batch_size=1)
    seq = make(0, 4, 3)
    sched.add_request(seq)
    sched.step()
    assert seq.status == SeqStatus.RUNNING


# review: a finished sequence must be marked FINISHED by the scheduler
def test_finished_sequence_is_marked_finished():
    sched = StaticScheduler(batch_size=1)
    seq = make(0, 4, 1)
    sched.add_request(seq)
    run_one_step(sched)
    assert seq.status == SeqStatus.FINISHED
