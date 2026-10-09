# Phase 1 — Memory (problems 1–5)

Pure Python, no torch. Everything here is bookkeeping: no actual K/V tensors
exist yet.

---

## P1 — Block pool
**File:** `minivllm/core/block_pool.py`
```python
class BlockPool:
    def __init__(self, num_blocks: int): ...
    def allocate(self) -> int: ...          # raises NoFreeBlocks when empty
    def free(self, block_id: int) -> None:  # raises on double-free / unknown id
    @property
    def num_free(self) -> int: ...
```
**Constraints:** allocate and free O(1).
**Tests to write:** allocate all then one more raises; free then reallocate
works; double-free raises; freeing out-of-range id raises; num_free tracks
correctly through a random sequence of 10k ops.
**Checkpoint:** Why fixed-size blocks instead of malloc-style variable sizes?
What does that trade away?
**vLLM ref (read after):** `vllm/v1/core/block_pool.py` (free block queue).

---

## P2 — Slot mapping
**File:** `minivllm/core/block_table.py`
```python
def slot_for_position(block_table: list[int], pos: int, block_size: int) -> int: ...
def slot_mapping(block_table: list[int], start: int, end: int, block_size: int) -> list[int]: ...
```
**Tests to write:** table `[5, 2, 9]`, block_size 16: pos 0, 15, 16, 37, 47;
pos beyond table raises; slot_mapping over a range crossing two block
boundaries; block_size 1 degenerate case.
**Checkpoint:** What is the slot of token 37 with table `[5, 2, 9]` and block
size 16? Why does attention need slots and not just block ids?

---

## P3 — append_token / can_append
**File:** `minivllm/core/block_table.py`
```python
class BlockTable:
    def __init__(self, pool: BlockPool, block_size: int): ...
    def num_blocks_needed(self, num_new_tokens: int) -> int: ...
    def can_append(self, num_new_tokens: int) -> bool: ...
    def append_tokens(self, num_new_tokens: int) -> None: ...   # allocates as needed
    def free_all(self) -> None: ...
    @property
    def num_tokens(self) -> int: ...
```
**Tests to write:** appending 1 token at a time allocates a new block exactly
at token 0, 16, 32; appending 40 at once allocates 3; can_append false when
pool can't cover it and append does not partially allocate (atomic);
free_all returns every block.
**Checkpoint:** At most how many slots can one sequence waste, and where are
they? Why must append be all-or-nothing?
**vLLM ref:** `vllm/v1/core/kv_cache_manager.py` (`allocate_slots`),
`single_type_kv_cache_manager.py`.

---

## P4 — Contiguous baseline + utilization
**File:** `minivllm/core/contiguous.py`
```python
class ContiguousAllocator:          # reserves max_len slots per sequence up front
    def __init__(self, total_slots: int, max_len: int): ...
    def can_admit(self) -> bool: ...
    def admit(self, seq_id: int) -> None: ...
    def append(self, seq_id: int, n: int) -> None: ...
    def release(self, seq_id: int) -> None: ...
def utilization(used_slots: int, reserved_slots: int) -> float: ...
```
**Tests to write:** with total 4096 slots and max_len 1024 only 4 sequences
admit regardless of actual length; utilization of paged vs contiguous on a
fixed length list matches a hand-computed value.
**Checkpoint:** Why does the paged-vs-contiguous gap grow with output-length
variance? When would contiguous be fine?

---

## P5 — Sequence state machine
**File:** `minivllm/core/sequence.py`
```python
class SeqStatus(Enum): WAITING, RUNNING, PREEMPTED, FINISHED
@dataclass
class Sequence:
    seq_id: int; prompt_len: int; max_new_tokens: int; arrival_time: float
    num_computed_tokens: int = 0; output_len: int = 0
    def num_uncomputed(self) -> int: ...
    def is_prefill(self) -> bool: ...
    def on_tokens_computed(self, n: int, sampled: bool) -> None: ...
    def preempt_recompute(self) -> None: ...   # resets computed, keeps outputs
    def is_finished(self) -> bool: ...
```
**Tests to write:** illegal transitions raise (FINISHED -> RUNNING);
after preempt_recompute, num_uncomputed == prompt_len + output_len; a decode
step computes exactly 1 token; finishing at max_new_tokens.
**Checkpoint:** Why do prefill, chunked prefill, decode and recompute all
reduce to "process the uncomputed tokens"?
