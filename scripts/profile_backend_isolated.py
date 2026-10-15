"""Isolated timing of one paged_backend_forward call (one layer) at batch 1, 8 and 32, ctx 100.

Diagnostic only. Separates CPU enqueue time from GPU completion: if enqueue-only time equals
the synced time, the call is CPU-bound. Run from the repo root: python scripts/profile_backend_isolated.py
"""
import sys, time, statistics
from types import SimpleNamespace
sys.path.insert(0, ".")
import torch
from minivllm.attention.backend import SeqSlice, StepContext, paged_backend_forward
from minivllm.attention.kv_pool import KVPool
BLOCK, kvh, qh, hd, ctx, L = 16, 2, 14, 64, 100, 24
def run(batch):
    nb = -(-ctx // BLOCK)
    pool = KVPool(L, batch * nb + 1, BLOCK, kvh, hd, dtype=torch.float16, device="cuda")
    tables = [list(range(i * nb, (i + 1) * nb)) for i in range(batch)]
    last = ctx - 1
    slots = torch.tensor([t[last // BLOCK] * BLOCK + last % BLOCK for t in tables], dtype=torch.long, device="cuda")
    c = StepContext(pool=pool, slots=slots, seqs=[SeqSlice(i, 1, tables[i], ctx) for i in range(batch)])
    q = torch.randn(1, qh, batch, hd, dtype=torch.float16, device="cuda")
    k = torch.randn(1, kvh, batch, hd, dtype=torch.float16, device="cuda")
    v = torch.randn(1, kvh, batch, hd, dtype=torch.float16, device="cuda")
    mod = SimpleNamespace(layer_idx=0, sliding_window=None)
    for _ in range(3): paged_backend_forward(mod, q, k, v, None, kv_context=c)
    torch.cuda.synchronize()
    # enqueue-only: 20 calls back to back, one sync at the end
    t = time.perf_counter()
    for _ in range(20): paged_backend_forward(mod, q, k, v, None, kv_context=c)
    cpu_enqueue = (time.perf_counter() - t) / 20 * 1e3
    torch.cuda.synchronize()
    total_back_to_back = (time.perf_counter() - t) / 20 * 1e3
    # each call synced
    ts = []
    for _ in range(20):
        t = time.perf_counter(); paged_backend_forward(mod, q, k, v, None, kv_context=c); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return dict(batch=batch, enqueue_only_ms=round(cpu_enqueue, 3), back_to_back_ms=round(total_back_to_back, 3), synced_each_call_ms=round(statistics.median(ts), 3))
for b in (1, 8, 32):
    print(run(b), flush=True)
