"""RealRunner.forward(sample_rows=...) returns the requested rows of the full logits, in the order given.

lm_head runs only on the sampled hidden rows, so the engine's sampled tokens must not change.
"""

import torch

from engine_helpers import make_models
from minivllm.attention.backend import SeqSlice
from minivllm.attention.kv_pool import KVPool
from minivllm.runner.real import RealRunner


def _runner(device):
    _, paged = make_models(device=device)
    cfg = paged.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    pool = KVPool(cfg.num_hidden_layers, 4, 16, cfg.num_key_value_heads, head_dim, dtype=torch.float64, device=device)
    return RealRunner(paged, pool)


def test_sample_rows_return_those_rows_of_the_full_logits(device):
    runner = _runner(device)
    tokens = [3, 17, 42, 8, 31] + [1, 4, 9, 16, 25, 36, 49, 2, 7]
    seqs = [SeqSlice(0, 5, [0], 5), SeqSlice(5, 9, [1], 9)]
    full = runner.forward(tokens, seqs)
    rows = [13, 4, 8]  # out of order, and one row from each sequence
    picked = runner.forward(tokens, seqs, sample_rows=rows)
    assert picked.shape == (len(rows), full.shape[-1])
    assert torch.allclose(picked, full[rows], atol=1e-10)


def test_sample_rows_on_a_decode_step_match_the_full_logits(device):
    runner = _runner(device)
    prefill = [3, 17, 42, 8, 31] + [1, 4, 9, 16, 25, 36, 49, 2, 7]
    runner.forward(prefill, [SeqSlice(0, 5, [0], 5), SeqSlice(5, 9, [1], 9)])
    tokens = [11, 23]
    seqs = [SeqSlice(0, 1, [0], 6), SeqSlice(1, 1, [1], 10)]  # two single-token rows: the batched decode path
    full = runner.forward(tokens, seqs)
    picked = runner.forward(tokens, seqs, sample_rows=[1, 0])
    assert torch.allclose(picked, full[[1, 0]], atol=1e-10)


def test_empty_sample_rows_give_no_logits_but_still_run(device):
    runner = _runner(device)
    tokens = [3, 17, 42]
    seqs = [SeqSlice(0, 3, [0], 3)]
    picked = runner.forward(tokens, seqs, sample_rows=[])
    assert picked.shape[0] == 0
    # The forward still wrote the KV: a following decode step sees the prompt.
    after = runner.forward([5], [SeqSlice(0, 1, [0], 4)])
    assert after.shape[0] == 1
