"""Hand-written greedy decode loop on a Hugging Face model, with a KV cache.

Problem 18 — Hand-written decode loop

Story
    generate() hides the decode loop. The engine has to run that loop itself:
    feed the prompt once, keep the cache the model returns, and feed back one
    token at a time. Once this matches generate() token for token, every later
    change (paged cache, batching) has a trusted baseline to compare against.

Operations
    greedy_generate(model, input_ids, max_new_tokens) -> torch.Tensor
        input_ids: [1, T] prompt token ids
        returns: [1, T + max_new_tokens], the prompt followed by the greedy tokens

Choices (the spec leaves these open)
    Greedy means taking the argmax of the last position's logits, with no sampling.
    The loop stops after exactly max_new_tokens tokens. It does not stop on an
    end-of-sequence token, which keeps the output length fixed.
    Batch size must be 1. Anything else raises ValueError.
    max_new_tokens == 0 returns the prompt unchanged.

Errors
    Raises ValueError when input_ids has more than one row, or when max_new_tokens < 0.

Example
    prompt of 5 tokens, max_new_tokens = 3
        the result has 8 tokens: the 5 prompt tokens, then 3 greedy tokens.
        Feeding the whole prompt once produces the first new token.
        Each later step feeds only the single newest token, plus the cache.

Constraints
    The prompt is fed in one forward pass. Each later step feeds exactly one token.
    The cache returned by the model is reused, not rebuilt from scratch.

Edge cases to think about
    What does the model return for the prompt, and what does the cache hold?
    Why is the logit taken from the last position only?
    What should happen with a one-token prompt?

Tests
    tests/test_p18_hf_decode.py checks token-for-token agreement with
    model.generate(do_sample=False) on a small random model, for a longer prompt,
    a one-token prompt, and the output length.
"""

from __future__ import annotations

import torch


def greedy_generate(model: torch.nn.Module, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy decode with the model's KV cache. See the module docstring.

    Raises ValueError when input_ids has more than one row, or max_new_tokens < 0.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, T], got {input_ids.shape}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if max_new_tokens == 0:
        return input_ids
    output_ids = input_ids.clone()
    # Feed the prompt once to get the first new token and the cache.
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        cache = outputs.past_key_values
        first_new_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, first_new_token], dim=-1)
    # Feed one token at a time, reusing the cache.
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(output_ids[:, -1:], past_key_values=cache)
            logits = outputs.logits
            cache = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=-1)
    return output_ids
    raise NotImplementedError
