"""Shared helpers for the engine tests: a float64 tiny Qwen2 pair, engine construction, and workloads.

Every engine built here runs the P13 invariant checker after each step.
"""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from minivllm.attention.backend import BACKEND_NAME, register_backend
from minivllm.clock import Clock
from minivllm.core.invariants import check
from minivllm.engine import Engine

BLOCK = 16
BASE = dict(
    vocab_size=50,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    eos_token_id=None,
    bos_token_id=None,
)


def make_models(seed: int = 0, device: str = "cpu"):
    register_backend()
    torch.manual_seed(seed)
    ref = Qwen2ForCausalLM(Qwen2Config(**BASE, attn_implementation="sdpa")).eval().double()
    paged = Qwen2ForCausalLM(Qwen2Config(**BASE, attn_implementation=BACKEND_NAME)).eval().double()
    paged.load_state_dict(ref.state_dict())
    return ref.to(device), paged.to(device)


def hf_generate(ref, prompt: list[int], n: int) -> list[int]:
    with torch.no_grad():
        out = ref.generate(torch.tensor([prompt], device=ref.device), do_sample=False, max_new_tokens=n, eos_token_id=None, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def make_engine(
    paged,
    num_blocks: int,
    max_batch: int,
    token_budget: int,
    prefix_cache: bool = False,
    clock: Clock | None = None,
    check_invariants: bool = True,
    batched_decode: bool = True,
) -> Engine:
    cfg = paged.config
    engine = Engine(
        paged,
        num_blocks=num_blocks,
        block_size=BLOCK,
        max_batch=max_batch,
        token_budget=token_budget,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=torch.float64,
        prefix_cache=prefix_cache,
        clock=clock,
        batched_decode=batched_decode,
    )
    if not check_invariants:
        return engine
    # Every engine test runs the P13 invariant checker after each step.
    plain_step = engine.step

    def checked_step() -> list[int]:
        finished = plain_step()
        check(engine.scheduler)
        return finished

    engine.step = checked_step
    return engine


def prompts(lengths: list[int], seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 50, (n,), generator=g).tolist() for n in lengths]


def shared_prefix_workload(seed: int = 2) -> list[list[int]]:
    """Three prompts that share a 32-token (two-block) prefix and differ in their tails."""
    g = torch.Generator().manual_seed(seed)
    shared = torch.randint(1, 50, (32,), generator=g).tolist()
    return [shared + torch.randint(1, 50, (n,), generator=g).tolist() for n in (3, 5, 2)]


def run_tracking_peak(engine: Engine, reqs: list[list[int]], max_new: int) -> tuple[dict[int, list[int]], int]:
    """Run every request and return (outputs, peak number of blocks held from the pool)."""
    for i, p in enumerate(reqs):
        engine.add_request(i, p, max_new)
    peak = 0
    while engine.has_work():
        engine.step()
        peak = max(peak, engine.block_pool.num_blocks - engine.block_pool.num_free)
    return {i: engine.tokens[i][len(reqs[i]):] for i in range(len(reqs))}, peak
