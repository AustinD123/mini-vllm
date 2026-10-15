"""RealRunner: runs one packed step of a Hugging Face model through the paged backend.

Scaffold for problem 23. The agent wrote this shell. The attention itself lives in
minivllm/attention/backend.py, which the human implements.

Operations
    RealRunner(model, pool)
        model  a Hugging Face model built with attn_implementation=BACKEND_NAME
        pool   the KVPool shared by every sequence
    forward(token_ids, seqs) -> torch.Tensor
        token_ids  list of token ids, packed in the order the SeqSlices lay them out
        seqs       list of SeqSlice covering the packed row
        returns    [T, vocab] logits, one row per packed token

Positions and slots are derived here, not by the caller:
    token i of a sequence sits at position context_len - length + i, and its slot
    comes from its block table.
"""

from __future__ import annotations

import torch

from minivllm.attention.backend import SeqSlice, StepContext
from minivllm.attention.kv_pool import KVPool
from minivllm.core.block_table import slot_mapping


class RealRunner:
    """Holds the model and pool, and runs packed steps. See the module docstring."""

    def __init__(self, model: torch.nn.Module, pool: KVPool) -> None:
        self.model = model
        self.pool = pool

    def forward(self, token_ids: list[int], seqs: list[SeqSlice]) -> torch.Tensor:
        """Run one packed step and return [T, vocab] logits.

        Every per-step tensor (input ids, positions, write slots, the context slot index) is built
        from host lists into one packed tensor, and moved to the device with one copy. The layers
        then read slices of it, so a step makes one host-to-device copy and no per-layer copies.
        """
        block_size = self.pool.k.shape[2]
        device = self.pool.k.device
        T = len(token_ids)
        positions: list[int] = []
        slots: list[int] = []
        ctx_index: list[int] = []
        for seq in seqs:
            first = seq.context_len - seq.length
            positions.extend(range(first, seq.context_len))
            slots.extend(slot_mapping(seq.block_table, first, seq.context_len, block_size))
            ctx_index.extend(slot_mapping(seq.block_table, 0, seq.context_len, block_size))
        packed = torch.tensor(token_ids + positions + slots + ctx_index, dtype=torch.long).to(device)
        input_ids = packed[0:T].unsqueeze(0)
        position_ids = packed[T:2 * T].unsqueeze(0)
        slot_tensor = packed[2 * T:3 * T]
        index_tensor = packed[3 * T:3 * T + len(ctx_index)]
        ctx = StepContext(pool=self.pool, slots=slot_tensor, seqs=seqs, ctx_index=index_tensor, validated=True)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False, kv_context=ctx)
        return outputs.logits[0]
