# Phase 4 — Real model + paged attention (problems 17–24)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (GQA, small enough for a laptop GPU or
Colab T4; fp16 on T4, no bf16). Tests use a tiny random-weight model in
float64 on CPU for exact matching, plus one slow real-model test.

Agent scaffolds: model loading, tokenizer, tiny test model fixture, the
`RealRunner` shell, attention-backend registration with HF.

The human has not written custom PyTorch before. P17–18 are the warm-up; hint
generously on tensor shapes, never on the logic.

---

## P17 — Hand-written causal attention
**File:** `minivllm/attention/reference.py`
```python
def causal_attention(q, k, v) -> torch.Tensor: ...  # [heads, T, dim] each
```
**Tests to write:** matches `F.scaled_dot_product_attention(is_causal=True)`
within 1e-5; T=1; non-square case (queries are the last Tq of Tk keys).
**Checkpoint:** Where exactly does the causal mask go, and what changes when
there are fewer queries than keys?

---

## P18 — Hand-written decode loop
**File:** `minivllm/runner/hf_loop.py`
Greedy generation with HF `model(...)` + `past_key_values`, no `generate()`.
**Tests to write:** token-for-token match with `model.generate(do_sample=False)`.
**Checkpoint:** What is in `past_key_values`, what shape, and why does decode
only feed one token?

---

## P19 — KV pool + scatter
**File:** `minivllm/attention/kv_pool.py`
```python
class KVPool:  # k, v: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None: ...
```
**Tests to write:** written values readable at expected (block, offset);
writing to slots spanning blocks; no Python loop per token (vectorized).
**Checkpoint:** Why one big preallocated tensor instead of per-sequence tensors?

---

## P20 — Paged attention, one sequence
```python
def paged_attention_single(q, pool, layer, block_table, context_len) -> torch.Tensor: ...
```
Gather this sequence's K/V through its block table, then attend.
**Tests to write:** equals contiguous attention on the same K/V; context_len
not a multiple of block size; shuffled (non-contiguous) block table.
**Checkpoint:** What does a real PagedAttention kernel avoid that your gather
does not?

---

## P21 — Packed batch
Concatenate tokens from several sequences (mix of prefill chunks and decodes)
into one `[1, T]` input with explicit `position_ids`; attention must keep each
query inside its own sequence.
**Tests to write:** batched output equals running each sequence alone;
mix of a 100-token prefill chunk and three decodes.
**Checkpoint:** Why does packing beat padding, and what has to be per-sequence?

---

## P22 — GQA
Handle `num_q_heads > num_kv_heads` (Qwen2.5-0.5B: 14 q heads, 2 kv heads —
verify from config).
**Tests to write:** matches repeat-interleaved reference.
**Checkpoint:** What does GQA do to KV bytes per token, and to how many
sequences fit?

---

## P23 — HF attention backend + exactness
**File:** `minivllm/attention/backend.py` (registration scaffolded by agent)
The function HF calls per layer: write new K/V to the pool at this step's
slots, then attend via block tables. Wire into `RealRunner`.
**Tests to write:** engine output token-for-token equal to HF `generate()`
under: batching, chunked prefill, prefix caching, recompute preemption.
**Checkpoint:** Walk one decode step from scheduler output to sampled token.

---

## P24 — KV pool sizing
```python
def kv_bytes_per_token(config, dtype) -> int: ...
def num_blocks_for_memory(free_bytes: int, config, dtype, block_size: int) -> int: ...
```
Use `head_dim` from config if present (not hidden/heads blindly).
**Tests to write:** hand-computed value for Qwen2.5-0.5B in fp16.
**Checkpoint:** How many tokens of KV fit on a T4 after loading the weights?
