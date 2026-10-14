import pytest
import torch

from engine_helpers import (
    BLOCK,
    hf_generate,
    make_engine,
    make_models,
    prompts,
    run_tracking_peak,
    shared_prefix_workload,
)
from minivllm.core.sequence import SeqStatus


def test_single_request_matches_generate(device):
    ref, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=4, max_batch=1, token_budget=64)
    prompt = [3, 17, 42, 8, 31]
    engine.add_request(0, prompt, 6)
    assert engine.run()[0] == hf_generate(ref, prompt, 6)


def test_mixed_workload_matches_generate_per_request(device):
    ref, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=4, max_batch=2, token_budget=8)
    reqs = prompts([5, 21, 9, 3])
    for i, p in enumerate(reqs):
        engine.add_request(i, p, 6)
    outputs = engine.run()
    for i, p in enumerate(reqs):
        assert outputs[i] == hf_generate(ref, p, 6), f"request {i}"


def test_preemption_happens_and_outputs_still_match(device):
    # Three requests that each cross a block boundary in a 3-block pool, so they must preempt.
    ref, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=3, max_batch=3, token_budget=16)
    reqs = prompts([10, 10, 10])
    for i, p in enumerate(reqs):
        engine.add_request(i, p, 12)
    outputs = engine.run()
    assert engine.num_preemptions > 0, "scenario did not preempt; adjust the pool size"
    for i, p in enumerate(reqs):
        assert outputs[i] == hf_generate(ref, p, 12), f"request {i}"


# extra edge case
def test_step_never_exceeds_token_budget(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=4, max_batch=2, token_budget=8)
    for i, p in enumerate(prompts([21, 19])):
        engine.add_request(i, p, 4)
    while engine.has_work():
        engine.step()
        assert engine.max_step_tokens <= 8


def test_prefix_caching_through_engine_matches_generate(device):
    ref, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=2, token_budget=64, prefix_cache=True)
    shared = list(range(10, 26))
    a, b = shared + [1, 2, 3], shared + [7, 8]
    engine.add_request(0, a, 4)
    engine.add_request(1, b, 4)
    outputs = engine.run()
    assert outputs[0] == hf_generate(ref, a, 4)
    assert outputs[1] == hf_generate(ref, b, 4)


def run_staged_peak(engine, reqs, max_new):
    """Like run_tracking_peak, but the later requests arrive after the first prefill has run.

    Blocks are shareable only once computed, so a shared prefix saves memory only for a
    request that comes after the first one's prefill.
    """
    engine.add_request(0, reqs[0], max_new)
    peak = 0
    engine.step()
    peak = max(peak, engine.block_pool.num_blocks - engine.block_pool.num_free)
    for i in range(1, len(reqs)):
        engine.add_request(i, reqs[i], max_new)
    while engine.has_work():
        engine.step()
        peak = max(peak, engine.block_pool.num_blocks - engine.block_pool.num_free)
    return {i: engine.tokens[i][len(reqs[i]):] for i in range(len(reqs))}, peak


def test_prefix_cache_on_and_off_give_same_tokens_with_fewer_blocks(device):
    ref, paged = make_models(device=device)
    reqs = shared_prefix_workload()
    off = make_engine(paged, num_blocks=16, max_batch=3, token_budget=64)
    on = make_engine(paged, num_blocks=16, max_batch=3, token_budget=64, prefix_cache=True)
    out_off, peak_off = run_staged_peak(off, reqs, 4)
    out_on, peak_on = run_staged_peak(on, reqs, 4)
    assert out_on == out_off
    for i, p in enumerate(reqs):
        assert out_on[i] == hf_generate(ref, p, 4), f"request {i}"
    assert peak_on < peak_off
    assert on.cached_tokens > 0


def test_fully_cached_prompt_still_recomputes_its_last_token(device):
    ref, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=8, max_batch=1, token_budget=64, prefix_cache=True)
    prompt = list(range(10, 42))  # exactly two full blocks
    engine.add_request(0, prompt, 3)
    engine.run()
    engine.add_request(1, prompt, 3)
    outputs = engine.run()
    # The second copy could reuse both blocks, but the block holding its last
    # prompt token must be recomputed for the logits, so only the first block hits.
    assert engine.scheduler.cached_by_seq[1] == BLOCK
    assert outputs[1] == hf_generate(ref, prompt, 3)


def test_preempted_sequence_gets_cache_hits_on_its_own_blocks(device):
    ref, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=4, max_batch=2, token_budget=64, prefix_cache=True)
    reqs = prompts([17, 17], seed=5)
    for i, p in enumerate(reqs):
        engine.add_request(i, p, 20)
    # Blocks each sequence held just before its most recent preemption,
    # and the blocks it holds again once re-admitted.
    held_before: dict[int, set[int]] = {}
    readmitted: dict[int, set[int]] = {}
    while engine.has_work():
        snapshot = {sid: set(t.block_ids) for sid, t in engine.scheduler.tables.items()}
        engine.step()
        for sid in engine.preempted_seq_ids:
            held_before.setdefault(sid, snapshot[sid])
        for sid in held_before:
            if sid not in readmitted and engine.seqs[sid].status == SeqStatus.RUNNING:
                readmitted[sid] = set(engine.scheduler.tables[sid].block_ids)
    assert engine.num_preemptions > 0, "scenario did not preempt; adjust the pool size"
    victim = engine.preempted_seq_ids[0]
    assert engine.scheduler.cached_by_seq[victim] >= BLOCK
    assert held_before[victim] & readmitted[victim], "re-admitted sequence did not reuse its own blocks"
    outputs = {i: engine.tokens[i][len(reqs[i]):] for i in range(len(reqs))}
    for i, p in enumerate(reqs):
        assert outputs[i] == hf_generate(ref, p, 20), f"request {i}"
