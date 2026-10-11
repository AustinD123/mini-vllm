import pytest

from minivllm.core.block_pool import BlockPool
from minivllm.core.scheduler import Scheduler, choose_victim
from minivllm.core.sequence import SeqStatus, Sequence


def make(seq_id: int, prompt_len: int, max_new: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new,
        arrival_time=float(seq_id),
    )


def run(sched: Scheduler, seqs: list[Sequence], watch_admission_of: int | None = None, max_steps: int = 500):
    """Drive the scheduler. Returns (preempted_ids in order, admitted_while_preempted)."""
    preempted_ids: list[int] = []
    admitted_while_preempted = False
    for _ in range(max_steps):
        if all(s.status == SeqStatus.FINISHED for s in seqs):
            break
        before = {s.seq_id: s.num_uncomputed() for s in seqs}
        out = sched.step()
        still_preempted = any(s.status == SeqStatus.PREEMPTED for s in seqs)
        if still_preempted and watch_admission_of is not None and watch_admission_of in out.num_tokens:
            admitted_while_preempted = True
        sampled = {sid: 0 for sid, n in out.num_tokens.items() if n == before[sid]}
        sched.update(out, sampled)
        for s in seqs:
            if s.status == SeqStatus.PREEMPTED and s.seq_id not in preempted_ids:
                preempted_ids.append(s.seq_id)
    return preempted_ids, admitted_while_preempted


def test_small_pool_preempts_and_every_request_still_finishes():
    seqs = [make(0, 16, 20), make(1, 16, 20)]
    sched = Scheduler(BlockPool(3), 16, max_batch=2, token_budget=1000)
    for s in seqs:
        sched.add_request(s)
    preempted, _ = run(sched, seqs)
    assert preempted, "expected at least one preemption"
    assert all(s.status == SeqStatus.FINISHED for s in seqs)


def test_victim_is_the_newest_running_sequence():
    seqs = [make(0, 16, 20), make(1, 16, 20)]
    sched = Scheduler(BlockPool(3), 16, max_batch=2, token_budget=1000)
    for s in seqs:
        sched.add_request(s)
    preempted, _ = run(sched, seqs)
    assert preempted and preempted[0] == 1


def test_request_that_can_never_fit_is_rejected():
    sched = Scheduler(BlockPool(2), 16, max_batch=2, token_budget=1000)
    with pytest.raises(ValueError):
        sched.add_request(make(0, 16, 100))


def test_no_new_admission_while_preempted_request_waits():
    seqs = [make(0, 16, 20), make(1, 16, 20), make(2, 1, 1)]
    sched = Scheduler(BlockPool(3), 16, max_batch=2, token_budget=1000)
    for s in seqs:
        sched.add_request(s)
    preempted, admitted_while_preempted = run(sched, seqs, watch_admission_of=2)
    assert preempted, "expected a preemption for this check to mean anything"
    assert admitted_while_preempted is False


# extra edge case
def test_choose_victim_returns_newest():
    a, b = make(0, 4, 1), make(1, 4, 1)
    assert choose_victim([a, b]) is b


# extra edge case
def test_choose_victim_empty_raises():
    with pytest.raises(ValueError):
        choose_victim([])
