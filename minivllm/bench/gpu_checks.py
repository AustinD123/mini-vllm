"""Checks the Colab notebook runs on a GPU: fp16 agreement with HF, memory pressure, KV sizing.

Plumbing, not a numbered problem.

Operations
    agreement(engine_outputs, reference_outputs) -> dict
        top1: fraction of generated positions where the two systems pick the same token
        first_divergence: per request, the first index where they differ (None when identical)
        positions: total generated positions compared
        Both systems run free (each feeds back its own tokens), so after a divergence the rest
        of that request can differ too. top1 therefore understates per-step agreement; read it
        together with first_divergence.
    teacher_forced_agreement(paged_model, prompts, reference_outputs, *, block_size=16) -> dict
        feeds the reference tokens through the paged model in one pass and compares the argmax
        at every generated position with the reference token. Each position sees the reference
        history, so one wrong token does not cascade. top1 is over all generated positions.
    run_engine(paged_model, prompts, output_lens, *, num_blocks, max_batch, token_budget,
               block_size=16) -> (outputs, num_preemptions)
        runs the engine with every request arriving at 0 and reports how often it preempted
    sizing_check(config, dtype, *, block_size=16, fraction=0.5) -> dict
        CUDA only. Predicts the block count from free memory (P24, num_blocks_for_memory),
        allocates that many KV blocks, and compares the bytes measured by
        torch.cuda.max_memory_allocated against the prediction.
"""

from __future__ import annotations

import torch

import math

from minivllm.attention.backend import SeqSlice
from minivllm.attention.kv_pool import KVPool
from minivllm.engine import Engine
from minivllm.runner.real import RealRunner
from minivllm.runner.sizing import kv_bytes_per_token, num_blocks_for_memory


def agreement(engine_outputs: dict[int, list[int]], reference_outputs: dict[int, list[int]]) -> dict:
    if engine_outputs.keys() != reference_outputs.keys():
        raise ValueError("both systems must answer the same request ids")
    matches = 0
    positions = 0
    first_divergence: dict[int, int | None] = {}
    for request_id in sorted(reference_outputs):
        got, want = engine_outputs[request_id], reference_outputs[request_id]
        if len(got) != len(want):
            raise ValueError(f"request {request_id}: {len(got)} vs {len(want)} tokens")
        positions += len(want)
        matches += sum(a == b for a, b in zip(got, want))
        first_divergence[request_id] = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    top1 = matches / positions if positions else 1.0
    return {"top1": top1, "first_divergence": first_divergence, "positions": positions}


def teacher_forced_agreement(
    paged_model,
    prompts: list[list[int]],
    reference_outputs: dict[int, list[int]],
    *,
    block_size: int = 16,
) -> dict:
    """Argmax of the paged model at each generated position, given the reference history.

    Logits at index i predict token i + 1, so the generated token k (at index len(prompt) + k)
    is predicted by the logits at index len(prompt) + k - 1.
    """
    cfg = paged_model.config
    param = next(paged_model.parameters())
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    matches = positions = 0
    per_request: dict[int, float] = {}
    for request_id in sorted(reference_outputs):
        prompt = list(prompts[request_id])
        generated = reference_outputs[request_id]
        tokens = prompt + generated
        total = len(tokens)
        blocks = math.ceil(total / block_size)
        pool = KVPool(
            cfg.num_hidden_layers, blocks, block_size, cfg.num_key_value_heads, head_dim,
            dtype=param.dtype, device=param.device,
        )
        runner = RealRunner(paged_model, pool)
        logits = runner.forward(tokens, [SeqSlice(0, total, list(range(blocks)), total)])
        predicted = logits[len(prompt) - 1:total - 1].argmax(dim=-1).tolist()
        hits = sum(p == g for p, g in zip(predicted, generated))
        per_request[request_id] = hits / len(generated) if generated else 1.0
        matches += hits
        positions += len(generated)
    top1 = matches / positions if positions else 1.0
    return {"top1": top1, "per_request": per_request, "positions": positions}


def run_engine(
    paged_model,
    prompts: list[list[int]],
    output_lens: list[int],
    *,
    num_blocks: int,
    max_batch: int,
    token_budget: int,
    block_size: int = 16,
) -> tuple[dict[int, list[int]], int]:
    cfg = paged_model.config
    engine = Engine(
        paged_model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch=max_batch,
        token_budget=token_budget,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    )
    for i, (prompt, n) in enumerate(zip(prompts, output_lens)):
        engine.add_request(i, list(prompt), n)
    outputs = engine.run()
    return outputs, engine.num_preemptions


def sizing_check(config, dtype: torch.dtype, *, block_size: int = 16, fraction: float = 0.5) -> dict:
    """Allocate the predicted KV pool on the GPU and compare measured bytes with the prediction.

    fraction is the share of currently free GPU memory given to the KV pool. The rest is left
    for activations and the allocator. Raises RuntimeError when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("sizing_check needs CUDA")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes * fraction)
    predicted_blocks = num_blocks_for_memory(budget, config, dtype, block_size)
    predicted_bytes = predicted_blocks * block_size * kv_bytes_per_token(config, dtype)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    pool = KVPool(
        config.num_hidden_layers,
        predicted_blocks,
        block_size,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
        dtype=dtype,
        device="cuda",
    )
    torch.cuda.synchronize()
    measured_peak = torch.cuda.max_memory_allocated() - before
    result = {
        "kv_memory_fraction": fraction,
        "predicted_blocks": predicted_blocks,
        "predicted_bytes": predicted_bytes,
        "pool_bytes": pool.k.numel() * pool.k.element_size() + pool.v.numel() * pool.v.element_size(),
        "measured_peak_bytes": measured_peak,
        "peak_over_prediction": measured_peak / predicted_bytes if predicted_bytes else float("nan"),
    }
    del pool
    torch.cuda.empty_cache()
    return result
