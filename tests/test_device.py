"""CUDA parity and dtype handling. The CUDA test is skipped when no GPU build of torch is installed."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from engine_helpers import BASE, make_engine, make_models
from minivllm.engine import Engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tiny_model_fp32_logits_on_cuda_match_cpu_within_1e4():
    torch.manual_seed(0)
    cpu_model = Qwen2ForCausalLM(Qwen2Config(**BASE, attn_implementation="sdpa")).eval()
    cuda_model = copy.deepcopy(cpu_model).to("cuda")
    ids = torch.tensor([[3, 17, 42, 8, 31, 9, 26]])
    with torch.no_grad():
        cpu_logits = cpu_model(ids).logits
        cuda_logits = cuda_model(ids.to("cuda")).logits.cpu()
    assert (cpu_logits - cuda_logits).abs().max().item() < 1e-4


def test_engine_gives_kv_pool_the_models_dtype_not_float32():
    _, paged = make_models()
    paged_fp16 = copy.deepcopy(paged).half()
    cfg = paged_fp16.config
    engine = Engine(
        paged_fp16,
        num_blocks=4,
        block_size=16,
        max_batch=1,
        token_budget=16,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    )
    assert engine.kv.k.dtype == torch.float16
    assert engine.kv.v.dtype == torch.float16


def test_engine_puts_kv_pool_on_the_models_device(device):
    _, paged = make_models(device=device)
    engine = make_engine(paged, num_blocks=4, max_batch=1, token_budget=16)
    assert engine.kv.k.device.type == torch.device(device).type


def test_sliding_window_config_is_rejected():
    model = Qwen2ForCausalLM(Qwen2Config(**BASE, attn_implementation="sdpa", use_sliding_window=True, sliding_window=8))
    cfg = model.config
    with pytest.raises(NotImplementedError):
        Engine(
            model,
            num_blocks=4,
            block_size=16,
            max_batch=1,
            token_budget=16,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
        )
