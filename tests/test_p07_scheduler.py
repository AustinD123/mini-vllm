import pytest

from minivllm.core.block_pool import BlockPool
from minivllm.core.scheduler import Scheduler
from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.core.sequence import SeqStatus, Sequence

BLOCK_SIZE = 16


def make(seq_id: int, prompt_len: int, max_new: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new,
        arrival_time=float(seq_id),
    )


def drive(sched: Scheduler) -> tuple[SchedulerOutput, list[Sequence]]:
    out = sched.step()
    sampled = {seq_id: 0 for seq_id in out.num_tokens}
    finished = sched.update(out, sampled)
    return out, finished


def test_request_arriving_mid_run_joins_next_step():
    sched = Scheduler(BlockPool(10), BLOCK_SIZE, max_batch=4, token_budget=1000)
    sched.add_request(make(0, 4, 5))
    drive(sched)
    sched.add_request(make(1, 4, 5))
    out, _ = drive(sched)
    assert set(out.num_tokens) == {0, 1}


def test_strict_fcfs_small_request_does_not_jump_ahead():
    sched = Scheduler(BlockPool(2), BLOCK_SIZE, max_batch=4, token_budget=1000)
    big = make(0, 40, 5)  # needs 3 blocks, pool has 2
    small = make(1, 4, 1)  # would fit, but is behind big
    sched.add_request(big)
    sched.add_request(small)
    out, _ = drive(sched)
    assert out.num_tokens == {}
    assert small.status == SeqStatus.WAITING


def test_finished_sequence_blocks_are_free_after_update():
    pool = BlockPool(4)
    sched = Scheduler(pool, BLOCK_SIZE, max_batch=2, token_budget=1000)
    sched.add_request(make(0, 20, 1))
    _, finished = drive(sched)
    assert [seq.seq_id for seq in finished] == [0]
    assert pool.num_free == 4


# extra edge case
def test_max_batch_limits_running_sequences():
    sched = Scheduler(BlockPool(10), BLOCK_SIZE, max_batch=2, token_budget=1000)
    for i in range(3):
        sched.add_request(make(i, 4, 1))
    out, _ = drive(sched)
    assert set(out.num_tokens) == {0, 1}


# extra edge case
def test_waiting_request_admitted_when_blocks_free():
    sched = Scheduler(BlockPool(2), BLOCK_SIZE, max_batch=4, token_budget=1000)
    sched.add_request(make(0, 20, 1))  # 2 blocks, uses the whole pool
    sched.add_request(make(1, 4, 1))  # 1 block
    out, _ = drive(sched)
    assert set(out.num_tokens) == {0}
    out, _ = drive(sched)
    assert set(out.num_tokens) == {1}


# extra edge case
def test_admitted_sequence_is_running_and_finished_status_set():
    sched = Scheduler(BlockPool(4), BLOCK_SIZE, max_batch=2, token_budget=1000)
    seq = make(0, 4, 1)
    sched.add_request(seq)
    out = sched.step()
    assert seq.status == SeqStatus.RUNNING
    sched.update(out, {0: 0})
    assert seq.status == SeqStatus.FINISHED


# extra edge case
def test_decode_step_computes_one_token():
    sched = Scheduler(BlockPool(4), BLOCK_SIZE, max_batch=2, token_budget=1000)
    sched.add_request(make(0, 4, 3))
    drive(sched)
    out, _ = drive(sched)
    assert out.num_tokens == {0: 1}


# extra edge case
def test_empty_scheduler_returns_no_work():
    sched = Scheduler(BlockPool(4), BLOCK_SIZE, max_batch=2, token_budget=1000)
    assert sched.step().num_tokens == {}


# extra edge case
def test_max_batch_below_one_raises():
    with pytest.raises(ValueError):
        Scheduler(BlockPool(4), BLOCK_SIZE, max_batch=0, token_budget=1000)
