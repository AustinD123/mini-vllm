"""Packed batch: several sequences in one [1, T] input, kept apart by an attention mask.

Problem 21 — Packed batch

Story
    Padding a batch to its longest sequence wastes compute on tokens that do
    nothing. Packing concatenates the sequences into one row, so the model runs
    once over real tokens only. Each token keeps its own position, and the
    attention mask has to stop a token from seeing any sequence but its own.

Operations
    pack_tokens(token_chunks, start_positions) -> (input_ids, position_ids, segment_ids)
        token_chunks: list of lists of token ids, one list per sequence
        start_positions: where each chunk starts inside its sequence
        input_ids:   [1, T]   all chunks concatenated, in order
        position_ids:[1, T]   each token's position inside its own sequence
        segment_ids: [T]      which chunk each token belongs to (0, 1, 2, ...)
    packed_attention_mask(segment_ids, position_ids) -> torch.Tensor
        returns a [T, T] bool tensor: entry (i, j) is True when query i may attend
        key j. That needs j in the same segment as i, and position_j <= position_i.

Choices (the spec leaves these open)
    Positions are taken from start_positions and count up within each chunk.
    The mask is returned as a boolean tensor; the caller converts it to the
    model's additive form.

Errors
    Raises ValueError when token_chunks is empty, or when the two lists have
    different lengths.

Example
    chunks [[1, 2, 3], [4, 5]], start_positions [0, 10]
        input_ids    [[1, 2, 3, 4, 5]]
        position_ids [[0, 1, 2, 10, 11]]
        segment_ids  [0, 0, 0, 1, 1]
        mask[3, 0] is False   (token 4 is in another sequence)
        mask[2, 0] is True    (same sequence, earlier position)
        mask[0, 2] is False   (token 1 may not see the future)

Constraints
    Each function runs in O(T) time, except the mask, which is O(T^2) by nature.
    No Python loop over tokens for the mask; build it with tensor operations.

Edge cases to think about
    What happens to the mask when one chunk is empty?
    Why does a query at position 10 still see key position 0 in its own sequence?
    Does the mask need the segment check if positions alone were unique?

Tests
    tests/test_p21_packed_batch.py covers a packed run matching each sequence run
    alone, a 100-token chunk packed with three short ones, and the mask's
    segment and causal rules.
"""

from __future__ import annotations

import torch


def pack_tokens(
    token_chunks: list[list[int]], start_positions: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate chunks into (input_ids [1, T], position_ids [1, T], segment_ids [T]).

    Raises ValueError when token_chunks is empty or the lists differ in length.
    """
    if len(token_chunks) == 0:
        raise ValueError("token_chunks is empty")
    if len(token_chunks) != len(start_positions):
        raise ValueError("token_chunks and start_positions must have the same length")

    input_ids = []
    position_ids = []
    segment_ids = []

    for segment_id, (chunk, start_pos) in enumerate(zip(token_chunks, start_positions)):
        input_ids.extend(chunk)
        position_ids.extend(range(start_pos, start_pos + len(chunk)))
        segment_ids.extend([segment_id] * len(chunk))

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([position_ids], dtype=torch.long),
        torch.tensor(segment_ids, dtype=torch.long),
    )


def packed_attention_mask(segment_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Return the [T, T] bool mask: same segment and key position <= query position."""
    if segment_ids.shape != position_ids.shape:
        raise ValueError("segment_ids and position_ids must have the same length")
    # Create a mask for same segment
    same_segment_mask = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)
    # Create a mask for causal attention (key position <= query position)
    causal_mask = position_ids.unsqueeze(0) <= position_ids.unsqueeze(1)
    # Combine both masks
    return same_segment_mask & causal_mask
