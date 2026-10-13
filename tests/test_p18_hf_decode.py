import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from minivllm.runner.hf_loop import greedy_generate


def make_model(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=2, n_head=2, eos_token_id=None)
    return GPT2LMHeadModel(config).eval()


def reference(model, input_ids: torch.Tensor, max_new: int) -> torch.Tensor:
    with torch.no_grad():
        return model.generate(input_ids, do_sample=False, max_new_tokens=max_new, eos_token_id=None, pad_token_id=0)


def test_matches_generate_token_for_token():
    model = make_model()
    prompt = torch.tensor([[3, 17, 42, 8, 91]])
    expected = reference(model, prompt, 10)
    with torch.no_grad():
        actual = greedy_generate(model, prompt, 10)
    assert torch.equal(actual, expected)


def test_one_token_prompt_matches_generate():
    model = make_model(seed=1)
    prompt = torch.tensor([[7]])
    expected = reference(model, prompt, 6)
    with torch.no_grad():
        actual = greedy_generate(model, prompt, 6)
    assert torch.equal(actual, expected)


def test_output_length_is_prompt_plus_new_tokens():
    model = make_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = greedy_generate(model, prompt, 5)
    assert out.shape == (1, 9)


# extra edge case
def test_zero_new_tokens_returns_prompt():
    model = make_model()
    prompt = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        out = greedy_generate(model, prompt, 0)
    assert torch.equal(out, prompt)


# extra edge case
def test_batch_larger_than_one_raises():
    model = make_model()
    prompt = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        greedy_generate(model, prompt, 2)
