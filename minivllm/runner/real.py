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

from minivllm.attention.backend import DecodePlan, SeqSlice, StepContext
from minivllm.attention.kv_pool import KVPool
from minivllm.core.block_table import slot_mapping


class RealRunner:
    """Holds the model and pool, and runs packed steps. See the module docstring."""

    def __init__(self, model: torch.nn.Module, pool: KVPool, batched_decode: bool = True) -> None:
        """batched_decode=False sends every sequence through the per-sequence path (the A/B baseline)."""
        self.model = model
        self.pool = pool
        self.batched_decode = batched_decode

    def forward(self, token_ids: list[int], seqs: list[SeqSlice], sample_rows: list[int] | None = None) -> torch.Tensor:
        """Run one packed step and return logits.

        sample_rows=None returns [T, vocab], one row per packed token. A list of packed row indices
        returns [len(sample_rows), vocab], in the order given. lm_head then runs only on those rows,
        so a prefill does not compute logits for every prompt token.

        Every per-step tensor (input ids, positions, write slots, the context slot index, the decode
        plan and the sample rows) is built from host lists into one packed tensor, and moved to the
        device with one copy. The layers then read slices of it, so a step makes one host-to-device
        copy and no per-layer copies.
        """
        block_size = self.pool.k.shape[2]
        device = self.pool.k.device
        T = len(token_ids)
        if T != sum(seq.length for seq in seqs):
            raise ValueError(f"{T} tokens for sequences that lay out {sum(seq.length for seq in seqs)} rows")
        positions: list[int] = []
        slots: list[int] = []
        ctx_index: list[int] = []
        decode = [seq for seq in seqs if seq.length == 1] if self.batched_decode else []
        for seq in seqs:
            first = seq.context_len - seq.length
            positions.extend(range(first, seq.context_len))
            slots.extend(slot_mapping(seq.block_table, first, seq.context_len, block_size))
            ctx_index.extend(slot_mapping(seq.block_table, 0, seq.context_len, block_size))
        # Decode plan: rows, context lengths and padded block tables, sent in the same copy.
        rows = [seq.start for seq in decode]
        lengths = [seq.context_len for seq in decode]
        width = max((-(-seq.context_len // block_size) for seq in decode), default=0)
        tables: list[int] = []
        for seq in decode:
            table = seq.block_table[:-(-seq.context_len // block_size)]
            tables.extend(table + [table[0]] * (width - len(table)))
        keep = list(sample_rows) if sample_rows is not None else []
        keep_offset = 3 * T + len(ctx_index) + len(rows) + len(lengths) + len(tables)
        packed = torch.tensor(token_ids + positions + slots + ctx_index + rows + lengths + tables + keep, dtype=torch.long).to(device)
        input_ids = packed[0:T].unsqueeze(0)
        position_ids = packed[T:2 * T].unsqueeze(0)
        slot_tensor = packed[2 * T:3 * T]
        start = 3 * T
        index_tensor = packed[start:start + len(ctx_index)]
        start += len(ctx_index)
        plan = None
        if decode:
            B = len(decode)
            plan = DecodePlan(
                rows=packed[start:start + B],
                context_lens=packed[start + B:start + 2 * B],
                block_tables=packed[start + 2 * B:start + 2 * B + B * width].view(B, width),
            )
        ctx = StepContext(pool=self.pool, slots=slot_tensor, seqs=seqs, ctx_index=index_tensor, validated=True, decode=plan)
        # A tensor logits_to_keep selects hidden rows before lm_head (transformers' Qwen2 forward).
        logits_to_keep = packed[keep_offset:keep_offset + len(keep)] if sample_rows is not None else 0
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False, kv_context=ctx,
                                 logits_to_keep=logits_to_keep)
        return outputs.logits[0]
