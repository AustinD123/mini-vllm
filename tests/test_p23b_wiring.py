"""Problem 23b wiring: the batched decode path and the per-sequence path give the same logits and outputs.

Both paths run the same model on the same float64 CPU weights. batched_decode=False is the A/B baseline.
"""

import torch

from engine_helpers import make_engine, make_models
from minivllm.attention.backend import SeqSlice
from minivllm.attention.kv_pool import KVPool
from minivllm.runner.real import RealRunner

BLOCK = 16


def _runner(paged, batched: bool, num_blocks: int = 8) -> RealRunner:
    cfg = paged.config
    pool = KVPool(cfg.num_hidden_layers, num_blocks, BLOCK, cfg.num_key_value_heads,
                  cfg.hidden_size // cfg.num_attention_heads, dtype=torch.float64)
    return RealRunner(paged, pool, batched_decode=batched)


def test_mixed_step_logits_match_with_and_without_batched_decode():
    _, paged = make_models()
    fast, slow = _runner(paged, True), _runner(paged, False)
    # Step 1: two prefills. Step 2: two decodes next to a new prefill chunk, with a block table
    # longer than the context needs. Step 3: decodes only.
    steps = [
        ([3, 9, 27, 4, 8, 1, 2, 6, 5], [
            SeqSlice(0, 5, [0, 1], 5), SeqSlice(5, 4, [2, 3], 4)]),
        ([11, 12, 13, 14, 15], [
            SeqSlice(0, 1, [0, 1], 6), SeqSlice(1, 1, [2, 3], 5), SeqSlice(2, 3, [4], 3)]),
        ([18, 19], [
            SeqSlice(0, 1, [0, 1, 6], 7), SeqSlice(1, 1, [2, 3], 6)]),
    ]
    for tokens, seqs in steps:
        got = fast.forward(tokens, seqs)
        want = slow.forward(tokens, seqs)
        assert torch.allclose(got, want, atol=1e-10), float((got - want).abs().max())


def test_engine_outputs_are_identical_with_and_without_batched_decode():
    _, paged = make_models()
    prompts = [[3, 17, 42, 8, 31], [5, 9], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
               [7, 7, 7], [40, 41, 42, 43, 44, 45, 46]]
    outputs = []
    for batched in (True, False):
        # A small token budget forces chunked prefill next to decodes.
        engine = make_engine(paged, num_blocks=24, max_batch=3, token_budget=10, batched_decode=batched)
        for i, prompt in enumerate(prompts):
            engine.add_request(i, list(prompt), 6)
        outputs.append(engine.run())
    assert outputs[0] == outputs[1]
    assert all(len(v) == 6 for v in outputs[0].values())
