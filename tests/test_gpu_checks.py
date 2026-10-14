import pytest
import torch
from transformers import Qwen2Config

from engine_helpers import hf_generate, make_models, prompts
from minivllm.bench.gpu_checks import agreement, sizing_check, teacher_forced_agreement


def test_agreement_reports_top1_and_first_divergence():
    got = {0: [1, 2, 3], 1: [4, 5]}
    want = {0: [1, 2, 9], 1: [4, 5]}
    result = agreement(got, want)
    assert result["positions"] == 5
    assert result["top1"] == pytest.approx(4 / 5)
    assert result["first_divergence"] == {0: 2, 1: None}


def test_agreement_rejects_mismatched_request_sets():
    with pytest.raises(ValueError):
        agreement({0: [1]}, {1: [1]})


@pytest.mark.skipif(torch.cuda.is_available(), reason="only checks the error path without CUDA")
def test_sizing_check_refuses_without_cuda():
    cfg = Qwen2Config(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    with pytest.raises(RuntimeError):
        sizing_check(cfg, torch.float16)


def test_teacher_forced_agreement_is_perfect_for_hf_greedy_outputs(device):
    ref, paged = make_models(device=device)
    reqs = prompts([5, 9, 3], seed=4)
    ref_out = {i: hf_generate(ref, p, 5) for i, p in enumerate(reqs)}
    result = teacher_forced_agreement(paged, reqs, ref_out)
    assert result["top1"] == 1.0
    assert result["positions"] == 15


def test_teacher_forced_agreement_drops_on_a_wrong_reference(device):
    ref, paged = make_models(device=device)
    reqs = prompts([5], seed=4)
    good = hf_generate(ref, reqs[0], 5)
    wrong = list(good)
    wrong[2] = (wrong[2] + 1) % 50  # the model's argmax at this position no longer matches
    result = teacher_forced_agreement(paged, reqs, {0: wrong})
    # Later positions are conditioned on the corrupted history, so they may also disagree; the score must still drop.
    assert result["top1"] < 1.0
    assert result["per_request"][0] < 1.0
