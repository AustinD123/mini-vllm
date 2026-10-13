import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from minivllm.runner.packed import pack_tokens, packed_attention_mask


def make_model(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(vocab_size=100, n_positions=256, n_embd=32, n_layer=2, n_head=2, eos_token_id=None, bos_token_id=0)
    return GPT2LMHeadModel(config).eval()


def mask_to_additive(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros(mask.shape, dtype=torch.float32).masked_fill(~mask, float("-inf"))[None, None]


def test_packed_logits_match_each_sequence_alone():
    model = make_model()
    chunks = [
        [int(x) for x in torch.randint(1, 100, (100,))],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    starts = [0, 0, 0, 0]
    ids, pos, seg = pack_tokens(chunks, starts)
    mask = packed_attention_mask(seg, pos[0])
    with torch.no_grad():
        packed = model(ids, position_ids=pos, attention_mask=mask_to_additive(mask)).logits[0]
        offset = 0
        for chunk in chunks:
            alone = model(torch.tensor([chunk])).logits[0]
            assert torch.allclose(packed[offset:offset + len(chunk)], alone, atol=1e-4)
            offset += len(chunk)


def test_pack_tokens_example():
    ids, pos, seg = pack_tokens([[1, 2, 3], [4, 5]], [0, 10])
    assert ids.tolist() == [[1, 2, 3, 4, 5]]
    assert pos.tolist() == [[0, 1, 2, 10, 11]]
    assert seg.tolist() == [0, 0, 0, 1, 1]


def test_mask_blocks_other_sequences_and_future_tokens():
    seg = torch.tensor([0, 0, 0, 1, 1])
    pos = torch.tensor([0, 1, 2, 10, 11])
    mask = packed_attention_mask(seg, pos)
    assert mask[3, 0].item() is False
    assert mask[2, 0].item() is True
    assert mask[0, 2].item() is False
    assert mask[3, 3].item() is True


# extra edge case
def test_query_at_later_position_still_sees_its_own_start():
    seg = torch.tensor([0, 0])
    pos = torch.tensor([0, 10])
    mask = packed_attention_mask(seg, pos)
    assert mask[1, 0].item() is True


# extra edge case
def test_empty_chunk_list_raises():
    with pytest.raises(ValueError):
        pack_tokens([], [])


# extra edge case
def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        pack_tokens([[1, 2]], [0, 5])


# review: segment_ids and position_ids of different lengths must raise ValueError
def test_mask_with_mismatched_lengths_raises_value_error():
    with pytest.raises(ValueError):
        packed_attention_mask(torch.tensor([0, 0, 0]), torch.tensor([0, 1]))
