"""Per-step overhead: one host-to-device copy per step, one sync for sampling, and no sync per layer.

The CUDA test counts synchronizing operations with torch's sync-debug mode. The count must not depend
on the batch size or the number of layers. The CPU test checks the range-check flag on KVPool.write.
"""

import warnings

import pytest
import torch

from engine_helpers import make_engine, make_models
from minivllm.attention.kv_pool import KVPool
from minivllm.core.block_table import slot_mapping


def test_write_without_range_check_stores_the_same_values():
    pool_a = KVPool(2, 4, 16, 2, 8)
    pool_b = KVPool(2, 4, 16, 2, 8)
    slots = torch.tensor([3, 17, 33], dtype=torch.long)
    k = torch.randn(3, 2, 8)
    v = torch.randn(3, 2, 8)
    pool_a.write(1, slots, k, v)
    pool_b.write(1, slots, k, v, check_range=False)
    assert torch.equal(pool_a.k, pool_b.k) and torch.equal(pool_a.v, pool_b.v)


def test_default_write_still_rejects_out_of_range_slot():
    pool = KVPool(1, 2, 16, 2, 8)
    with pytest.raises(ValueError):
        pool.write(0, torch.tensor([32], dtype=torch.long), torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))


def test_slot_mapping_is_still_the_reference_for_the_context_index():
    # The runner builds the context index with slot_mapping; this pins the shape it relies on.
    assert slot_mapping([5, 2], 0, 20, 16) == list(range(80, 96)) + list(range(32, 36))


def _decode_sync_count(batch: int) -> int:
    """Synchronizing ops in one steady-decode step with `batch` running sequences."""
    _, paged = make_models(device="cuda")
    engine = make_engine(paged, num_blocks=64, max_batch=batch, token_budget=max(64, batch * 2),
                         check_invariants=False)
    for i in range(batch):
        engine.add_request(i, [3 + i, 7, 11, 13], 40)
    for _ in range(200):
        run = engine.scheduler.running
        if len(run) == batch and all(s.num_uncomputed() == 1 for s in run):
            break
        engine.step()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            engine.step()
            torch.cuda.synchronize()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    return sum("synchronizing" in str(w.message) for w in seen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_decode_step_syncs_do_not_grow_with_batch_size():
    small = _decode_sync_count(2)
    large = _decode_sync_count(8)
    assert small == large
    assert large <= 4, f"a decode step made {large} synchronizing ops"
