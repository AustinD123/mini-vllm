"""Problem 23b: batched decode attention agrees with the one-sequence reference, and padding cannot leak.

The reference is paged_attention_single on the CPU, in float64. The batched call runs on the device the
`device` fixture gives (CPU, and CUDA when a GPU build of torch is installed).
"""

import json

import pytest
import torch

from minivllm.attention.batched import paged_decode_attention
from minivllm.attention.kv_pool import KVPool
from minivllm.attention.paged import paged_attention_single

BLOCK = 16


def build_case(num_heads: int, kv_heads: int, head_dim: int, context_lens: list[int], seed: int = 0):
    """A CPU float64 pool with random K and V, shuffled non-contiguous block tables, and the padded batch."""
    gen = torch.Generator().manual_seed(seed)
    blocks_per_seq = [-(-c // BLOCK) for c in context_lens]
    total = sum(blocks_per_seq)
    num_blocks = total + 5  # a few spare blocks that no sequence references
    pool = KVPool(2, num_blocks, BLOCK, kv_heads, head_dim, dtype=torch.float64)
    pool.k = torch.randn(pool.k.shape, generator=gen, dtype=torch.float64)
    pool.v = torch.randn(pool.v.shape, generator=gen, dtype=torch.float64)
    ids = torch.randperm(num_blocks, generator=gen).tolist()  # shuffled: no sequence owns contiguous blocks
    tables, cursor = [], 0
    for n in blocks_per_seq:
        tables.append(ids[cursor:cursor + n])
        cursor += n
    q = torch.randn(len(context_lens), num_heads, head_dim, generator=gen, dtype=torch.float64)
    return pool, tables, q


def pad(tables: list[list[int]]) -> torch.Tensor:
    width = max(len(t) for t in tables)
    return torch.tensor([t + [t[0]] * (width - len(t)) for t in tables], dtype=torch.long)


def expanded(pool: KVPool, group: int) -> KVPool:
    """A pool whose kv head h is the original kv head h // group. paged_attention_single (P20) has no GQA,
    so the reference expands the kv heads to match the query heads; the batched call never does."""
    out = KVPool(pool.k.shape[0], pool.k.shape[1], pool.k.shape[2], pool.k.shape[3] * group, pool.k.shape[4],
                 dtype=pool.k.dtype)
    out.k = pool.k.repeat_interleave(group, dim=3)
    out.v = pool.v.repeat_interleave(group, dim=3)
    return out


def reference(q, pool, tables, context_lens) -> torch.Tensor:
    """One paged_attention_single call per sequence on the expanded pool, the query at its last position."""
    group = q.shape[1] // pool.k.shape[3]
    ref_pool = expanded(pool, group)
    return torch.stack([
        paged_attention_single(q[b].unsqueeze(1), ref_pool, 0, tables[b], context_lens[b])[:, 0]
        for b in range(len(tables))
    ])


def on_device(pool: KVPool, device: str) -> KVPool:
    moved = KVPool(pool.k.shape[0], pool.k.shape[1], pool.k.shape[2], pool.k.shape[3], pool.k.shape[4],
                   dtype=pool.k.dtype, device=device)
    moved.k = pool.k.to(device)
    moved.v = pool.v.to(device)
    return moved


@pytest.mark.parametrize("context_lens", [[1, 15, 16, 17, 100, 300]])
def test_matches_single_sequence_reference_for_mixed_context_lengths(device, context_lens):
    pool, tables, q = build_case(4, 2, 8, context_lens)
    want = reference(q, pool, tables, context_lens)
    got = paged_decode_attention(
        q.to(device), on_device(pool, device), 0, pad(tables).to(device), torch.tensor(context_lens).to(device)
    )
    assert torch.allclose(got.cpu(), want, atol=1e-10)


def test_batch_of_one_matches_the_reference(device):
    pool, tables, q = build_case(4, 2, 8, [37], seed=3)
    want = reference(q, pool, tables, [37])
    got = paged_decode_attention(q.to(device), on_device(pool, device), 0, pad(tables).to(device),
                                 torch.tensor([37]).to(device))
    assert torch.allclose(got.cpu(), want, atol=1e-10)


def test_qwen_head_layout_gqa_matches_the_reference(device):
    # Qwen2.5-0.5B: 14 query heads share 2 kv heads (group size 7), head_dim 64.
    context_lens = [5, 16, 130]
    pool, tables, q = build_case(14, 2, 64, context_lens, seed=7)
    want = reference(q, pool, tables, context_lens)
    got = paged_decode_attention(q.to(device), on_device(pool, device), 0, pad(tables).to(device),
                                 torch.tensor(context_lens).to(device))
    assert got.shape == (3, 14, 64)
    assert torch.allclose(got.cpu(), want, atol=1e-10)


def test_nan_in_every_unreferenced_slot_leaves_the_output_unchanged(device):
    context_lens = [1, 17, 100]
    pool, tables, q = build_case(4, 2, 8, context_lens, seed=11)
    table = pad(tables)  # rows shorter than the widest one repeat their first block, which is used
    clean = paged_decode_attention(q.to(device), on_device(pool, device), 0, table.to(device),
                                   torch.tensor(context_lens).to(device))
    # NaN into every slot that no real position uses: spare blocks, and the unused tail of each last block.
    used = {(t[pos // BLOCK], pos % BLOCK) for t, n in zip(tables, context_lens) for pos in range(n)}
    k, v = pool.k.clone(), pool.v.clone()
    for block in range(pool.k.shape[1]):
        for offset in range(BLOCK):
            if (block, offset) not in used:
                k[:, block, offset] = float("nan")
                v[:, block, offset] = float("nan")
    poisoned = KVPool(2, pool.k.shape[1], BLOCK, 2, 8, dtype=torch.float64)
    poisoned.k, poisoned.v = k, v
    got = paged_decode_attention(q.to(device), on_device(poisoned, device), 0, table.to(device),
                                 torch.tensor(context_lens).to(device))
    assert torch.isfinite(got).all()
    assert torch.equal(got.cpu(), clean.cpu())


def _kernel_launches(fn, path) -> int:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    return sum(1 for e in events if e.get("cat") == "kernel")


def test_kernel_launches_do_not_grow_with_batch_size(device, tmp_path):
    """The count is the same for every batch size from 2 to 64. B=1 is not compared: cuBLAS runs a
    matrix-vector kernel there instead of a matrix-matrix one, which adds one launch. No Python loop
    over sequences is involved either way."""
    if device != "cuda":
        pytest.skip("kernel counts are a CUDA property")
    context_lens = [40] * 64
    pool, tables, q = build_case(14, 2, 64, context_lens, seed=5)
    cuda_pool = on_device(pool, "cuda")
    table = pad(tables).to("cuda")
    lengths = torch.tensor(context_lens, device="cuda")
    counts = {}
    for batch in (2, 32, 64):
        counts[batch] = _kernel_launches(
            lambda b=batch: paged_decode_attention(q[:b].to("cuda"), cuda_pool, 0, table[:b], lengths[:b]),
            tmp_path / f"b{batch}.json",
        )
    assert len(set(counts.values())) == 1, f"kernels per call by batch size: {counts}"
