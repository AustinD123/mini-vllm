import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from minivllm.attention.backend import BACKEND_NAME, SeqSlice, StepContext, paged_backend_forward, register_backend
from minivllm.attention.kv_pool import KVPool
from minivllm.runner.real import RealRunner

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


def make_models(seed: int = 0) -> tuple[Qwen2ForCausalLM, Qwen2ForCausalLM]:
    """Return (reference, paged) with identical weights. Reference uses HF sdpa, paged uses our backend."""
    register_backend()
    torch.manual_seed(seed)
    ref = Qwen2ForCausalLM(Qwen2Config(**BASE, attn_implementation="sdpa")).eval().double()
    paged = Qwen2ForCausalLM(Qwen2Config(**BASE, attn_implementation=BACKEND_NAME)).eval().double()
    paged.load_state_dict(ref.state_dict())
    return ref, paged


def make_runner(paged: Qwen2ForCausalLM, num_blocks: int = 8) -> RealRunner:
    cfg = paged.config
    pool = KVPool(cfg.num_hidden_layers, num_blocks, BLOCK, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads)
    pool.k = pool.k.double()
    pool.v = pool.v.double()
    return RealRunner(paged, pool)


def hf_logits(ref: Qwen2ForCausalLM, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return ref(torch.tensor([tokens])).logits[0]


def hf_generate(ref: Qwen2ForCausalLM, prompt: list[int], n: int) -> list[int]:
    with torch.no_grad():
        out = ref.generate(torch.tensor([prompt]), do_sample=False, max_new_tokens=n, eos_token_id=None, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def greedy(runner: RealRunner, prompt: list[int], table: list[int], n: int) -> list[int]:
    """Prefill the prompt in one step, then decode n tokens one at a time."""
    logits = runner.forward(prompt, [SeqSlice(0, len(prompt), table, len(prompt))])
    out = [int(logits[-1].argmax())]
    for _ in range(n - 1):
        logits = runner.forward([out[-1]], [SeqSlice(0, 1, table, len(prompt) + len(out))])
        out.append(int(logits[-1].argmax()))
    return out


def test_prefill_logits_match_hf():
    ref, paged = make_models()
    runner = make_runner(paged)
    prompt = [3, 17, 42, 8, 31]
    logits = runner.forward(prompt, [SeqSlice(0, 5, [0], 5)])
    assert torch.allclose(logits, hf_logits(ref, prompt), atol=1e-8)


def test_greedy_matches_generate_token_for_token():
    ref, paged = make_models()
    runner = make_runner(paged)
    prompt = [3, 17, 42, 8, 31]
    assert greedy(runner, prompt, [0], 8) == hf_generate(ref, prompt, 8)


def test_greedy_across_block_boundary_with_shuffled_table():
    ref, paged = make_models()
    runner = make_runner(paged)
    prompt = list(range(1, 22))  # 21 tokens: crosses the first block boundary
    assert greedy(runner, prompt, [5, 2], 10) == hf_generate(ref, prompt, 10)


def test_chunked_prefill_matches_single_prefill():
    ref, paged = make_models()
    runner = make_runner(paged)
    prompt = list(range(4, 24))  # 20 tokens
    table = [0, 1]
    consumed = 0
    logits = None
    for size in (7, 7, 6):
        chunk = prompt[consumed:consumed + size]
        consumed += size
        logits = runner.forward(chunk, [SeqSlice(0, size, table, consumed)])
    assert torch.allclose(logits[-1], hf_logits(ref, prompt)[-1], atol=1e-8)


def test_shared_prefix_block_is_read_not_rewritten():
    ref, paged = make_models()
    runner = make_runner(paged)
    shared = list(range(10, 26))  # exactly one full block
    prompt_a = shared + [1, 2, 3, 4]
    prompt_b = shared + [7, 8, 9]
    runner.forward(prompt_a, [SeqSlice(0, len(prompt_a), [0, 1], len(prompt_a))])
    # B reuses block 0 from A and writes only its own three tokens into block 2.
    logits = runner.forward(prompt_b[16:], [SeqSlice(0, 3, [0, 2], len(prompt_b))])
    assert torch.allclose(logits[-1], hf_logits(ref, prompt_b)[-1], atol=1e-8)


def test_recompute_after_preemption_matches_generate():
    ref, paged = make_models()
    runner = make_runner(paged)
    prompt = [5, 9, 21, 33, 2]
    first = greedy(runner, prompt, [0], 4)
    # Preempted: the old blocks are gone. Recompute prompt + generated tokens into new blocks.
    second = greedy(runner, prompt + first, [3], 4)
    assert first + second == hf_generate(ref, prompt, 8)


def test_packed_batch_matches_each_sequence_alone():
    ref, paged = make_models()
    runner = make_runner(paged)
    prompt_a = [3, 17, 42, 8, 31]
    prompt_b = [1, 4, 9, 16, 25, 36, 49, 2, 7]
    seqs = [SeqSlice(0, 5, [0], 5), SeqSlice(5, 9, [1], 9)]
    logits = runner.forward(prompt_a + prompt_b, seqs)
    assert torch.allclose(logits[:5], hf_logits(ref, prompt_a), atol=1e-8)
    assert torch.allclose(logits[5:], hf_logits(ref, prompt_b), atol=1e-8)
    # One packed decode step: each sequence feeds its own greedy token.
    tok_a = int(logits[4].argmax())
    tok_b = int(logits[-1].argmax())
    seqs = [SeqSlice(0, 1, [0], 6), SeqSlice(1, 1, [1], 10)]
    logits = runner.forward([tok_a, tok_b], seqs)
    assert torch.allclose(logits[0], hf_logits(ref, prompt_a + [tok_a])[-1], atol=1e-8)
    assert torch.allclose(logits[1], hf_logits(ref, prompt_b + [tok_b])[-1], atol=1e-8)


# extra edge case
def test_missing_kv_context_raises_value_error():
    _, paged = make_models()
    with pytest.raises(ValueError):
        with torch.no_grad():
            paged(torch.tensor([[1, 2, 3]]), use_cache=False)


class FakeModule(torch.nn.Module):
    """Just enough of an attention module for a direct call: a layer index, no sliding window."""

    def __init__(self, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx


def direct_inputs(t: int = 3, hq: int = 4, hkv: int = 2, d: int = 8):
    torch.manual_seed(0)
    q = torch.randn(1, hq, t, d)
    k = torch.randn(1, hkv, t, d)
    v = torch.randn(1, hkv, t, d)
    return q, k, v


# review: a sliding-window layer must fail loudly, not give wrong logits
def test_sliding_window_layer_raises_not_implemented():
    register_backend()
    cfg = Qwen2Config(
        **BASE,
        attn_implementation=BACKEND_NAME,
        use_sliding_window=True,
        sliding_window=4,
        layer_types=["full_attention", "sliding_attention"],
    )
    paged = Qwen2ForCausalLM(cfg).eval().double()
    runner = make_runner(paged)
    with pytest.raises(NotImplementedError):
        runner.forward([1, 2, 3], [SeqSlice(0, 3, [0], 3)])


# review: a non-default scaling must fail loudly, not be silently replaced by 1 / sqrt(head_dim)
def test_non_default_scaling_raises_not_implemented():
    pool = KVPool(1, 4, BLOCK, 2, 8)
    q, k, v = direct_inputs()
    ctx = StepContext(pool=pool, slots=torch.tensor([0, 1, 2]), seqs=[SeqSlice(0, 3, [0], 3)])
    with pytest.raises(NotImplementedError):
        paged_backend_forward(FakeModule(), q, k, v, None, scaling=1.0, kv_context=ctx)


# review: a 4-D attention_mask (which the backend ignores) must not be rejected
def test_four_dim_attention_mask_is_ignored():
    pool_a = KVPool(1, 4, BLOCK, 2, 8)
    pool_b = KVPool(1, 4, BLOCK, 2, 8)
    q, k, v = direct_inputs()
    slots = torch.tensor([0, 1, 2])
    seqs = [SeqSlice(0, 3, [0], 3)]
    out_a, _ = paged_backend_forward(FakeModule(), q, k, v, None, kv_context=StepContext(pool_a, slots, seqs))
    mask = torch.zeros(1, 1, 3, 3)
    out_b, _ = paged_backend_forward(FakeModule(), q, k, v, mask, kv_context=StepContext(pool_b, slots, seqs))
    assert torch.allclose(out_a, out_b)


# review: a block table too short for context_len must raise ValueError, not IndexError
def test_short_block_table_raises_value_error():
    pool = KVPool(1, 4, BLOCK, 2, 8)
    q, k, v = direct_inputs()
    ctx = StepContext(pool=pool, slots=torch.tensor([0, 1, 2]), seqs=[SeqSlice(0, 3, [0], 20)])
    with pytest.raises(ValueError):
        paged_backend_forward(FakeModule(), q, k, v, None, kv_context=ctx)


# review: q, k, v and the pool must be on the same device; checked before any write
def test_device_mismatch_raises_value_error():
    pool = KVPool(1, 4, BLOCK, 2, 8)
    q, k, v = direct_inputs()
    q_other = torch.empty_like(q, device="meta")
    ctx = StepContext(pool=pool, slots=torch.tensor([0, 1, 2]), seqs=[SeqSlice(0, 3, [0], 3)])
    with pytest.raises(ValueError):
        paged_backend_forward(FakeModule(), q_other, k, v, None, kv_context=ctx)


@pytest.mark.skipif(
    os.environ.get("MINIVLLM_REAL_MODEL") is None,
    reason="slow: set MINIVLLM_REAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct to run against real weights",
)
def test_real_model_greedy_matches_generate():
    from transformers import AutoTokenizer

    register_backend()
    name = os.environ["MINIVLLM_REAL_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    ref = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", torch_dtype=torch.float32).eval()
    paged = AutoModelForCausalLM.from_pretrained(name, attn_implementation=BACKEND_NAME, torch_dtype=torch.float32).eval()
    cfg = AutoConfig.from_pretrained(name)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    pool = KVPool(cfg.num_hidden_layers, 8, BLOCK, cfg.num_key_value_heads, head_dim)
    runner = RealRunner(paged, pool)
    prompt = tokenizer("The capital of France is", return_tensors="pt")["input_ids"][0].tolist()
    assert greedy(runner, prompt, [0, 1], 8) == hf_generate(ref, prompt, 8)
