"""Engine loop: scheduler decides, RealRunner computes, greedy sampling feeds back.

Scaffold written by the agent (plumbing, not a numbered problem).

Operations
    Engine(model, num_blocks, block_size, max_batch, token_budget,
           num_layers, num_kv_heads, head_dim, dtype)
        builds the BlockPool, the KVPool, the Scheduler and the RealRunner
    add_request(seq_id, prompt_ids, max_new_tokens) -> None
        queues a request with the scheduler and remembers its tokens
    step() -> list[int]
        one scheduler step: plan, run the model, sample greedily, update.
        Returns the seq_ids that finished this step.
    run() -> dict[int, list[int]]
        steps until no request is waiting or running; returns each request's output tokens
    num_preemptions
        how many times a running sequence was preempted (recompute path)

How one step works
    The scheduler says how many tokens each sequence computes this step. The engine
    takes those tokens from the sequence's full token list, starting at
    num_computed_tokens (after a preemption that is the start of the prompt again).
    A sequence gets a new output token only when this step computes everything it
    has, so its last row of logits is sampled.
"""

from __future__ import annotations

import heapq
import itertools
import math

import torch

from minivllm.attention.backend import SeqSlice
from minivllm.attention.kv_pool import KVPool
from minivllm.bench.metrics import RequestLog
from minivllm.clock import Clock, FakeClock
from minivllm.core.block_pool import BlockPool
from minivllm.core.prefix_cache import PrefixCache
from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import Sequence
from minivllm.runner.fake import step_latency
from minivllm.runner.real import RealRunner


def _uses_sliding_window(config) -> bool:
    """True when the config turns on sliding-window attention.

    Qwen2 sets use_sliding_window explicitly. Other configs (Mistral) only set sliding_window.
    """
    if getattr(config, "use_sliding_window", None):
        return True
    return getattr(config, "use_sliding_window", None) is None and bool(getattr(config, "sliding_window", None))


class Engine:
    """Ties the scheduler to the real model. See the module docstring."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_blocks: int,
        block_size: int,
        max_batch: int,
        token_budget: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype | None = None,
        prefix_cache: bool = False,
        clock: Clock | None = None,
        batched_decode: bool = True,
    ) -> None:
        """Build the engine. dtype and device default to the model's, so a fp16 model gets an fp16 KV pool.

        Raises NotImplementedError for a config that uses sliding-window attention.
        """
        if _uses_sliding_window(model.config):
            raise NotImplementedError("sliding-window attention is not supported by the paged engine")
        param = next(model.parameters())
        dtype = param.dtype if dtype is None else dtype
        self.block_size = block_size
        self.clock: Clock = clock if clock is not None else FakeClock()
        # Requests not yet handed to the scheduler, as a heap of (arrival_time, order, seq).
        self.pending: list[tuple[float, int, Sequence]] = []
        self._arrival_order = itertools.count()
        self.block_pool = BlockPool(num_blocks)
        self.kv = KVPool(num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=param.device)
        self.runner = RealRunner(model, self.kv, batched_decode=batched_decode)
        cache = PrefixCache(self.block_pool, block_size) if prefix_cache else None
        self.seqs: dict[int, Sequence] = {}
        self.tokens: dict[int, list[int]] = {}
        self.scheduler = Scheduler(
            self.block_pool, block_size, max_batch, token_budget, prefix_cache=cache, tokens=self.tokens
        )
        self.logs: dict[int, RequestLog] = {}
        self.num_preemptions = 0
        self.preempted_seq_ids: list[int] = []
        # KV use, in blocks. In use: held by running sequences. Occupancy: everything not free,
        # including cached blocks that no running sequence holds (the retained LRU).
        self.peak_in_use_blocks = 0
        self.peak_occupancy_blocks = 0
        self.max_step_tokens = 0

    @property
    def cached_tokens(self) -> int:
        """Prompt tokens served from the prefix cache, summed over every admission."""
        return self.scheduler.cached_tokens

    def add_request(self, seq_id: int, prompt_ids: list[int], max_new_tokens: int, arrival_time: float = 0.0) -> None:
        """Queue one request. Its tokens start as the prompt. It joins the scheduler at arrival_time.

        Raises ValueError, before any state changes, when the request is empty, asks for no
        output, cannot fit in the KV pool, or has a non-finite arrival time. A request whose
        arrival has already passed is handed to the scheduler before this call returns.
        """
        if not prompt_ids:
            raise ValueError("a request needs at least one prompt token")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
        capacity = self.block_pool.num_blocks * self.block_size
        if len(prompt_ids) + max_new_tokens > capacity:
            raise ValueError(
                f"request {seq_id} needs {len(prompt_ids) + max_new_tokens} tokens; the pool holds {capacity}"
            )
        if not math.isfinite(arrival_time):
            raise ValueError(f"arrival_time must be finite, got {arrival_time}")
        seq = Sequence(
            seq_id=seq_id, prompt_len=len(prompt_ids), max_new_tokens=max_new_tokens, arrival_time=arrival_time
        )
        self.seqs[seq_id] = seq
        self.tokens[seq_id] = list(prompt_ids)
        self.logs[seq_id] = RequestLog(seq_id, arrival_time, math.nan, [], math.nan)
        heapq.heappush(self.pending, (arrival_time, next(self._arrival_order), seq))
        self._release_due()

    def _release_due(self) -> None:
        """Hand every pending request whose arrival has passed to the scheduler, in arrival order."""
        now = self.clock.now()
        while self.pending and self.pending[0][0] <= now:
            _, _, seq = heapq.heappop(self.pending)
            self.scheduler.add_request(seq)

    def _release_arrived(self) -> None:
        """Release due requests. When nothing is waiting or running, first jump the clock to the next arrival.

        The jump and the release happen in the same call, so an idle gap costs no extra step.
        """
        if self.pending and not (self.scheduler.waiting or self.scheduler.running):
            self.clock.advance_to(self.pending[0][0])
        self._release_due()

    def has_work(self) -> bool:
        return bool(self.pending or self.scheduler.waiting or self.scheduler.running)

    def step(self) -> list[int]:
        """Run one scheduler step. Returns the seq_ids that finished in it."""
        if self.pending:
            self._release_arrived()
        running_before = {seq.seq_id for seq in self.scheduler.running}
        out = self.scheduler.step()
        preempted = sorted(running_before & {seq.seq_id for seq in self.scheduler.waiting})
        self.num_preemptions += len(preempted)
        self.preempted_seq_ids.extend(preempted)
        if not out.num_tokens:
            return []
        step_start = self.clock.now()
        self.max_step_tokens = max(self.max_step_tokens, sum(out.num_tokens.values()))

        token_ids: list[int] = []
        slices: list[SeqSlice] = []
        sample_rows: dict[int, int] = {}
        prefill_tokens = decode_seqs = context = 0
        start = 0
        for seq_id, n in out.num_tokens.items():
            seq = self.seqs[seq_id]
            before = seq.num_computed_tokens
            if math.isnan(self.logs[seq_id].dequeue_time):  # first step that schedules this request
                self.logs[seq_id].dequeue_time = step_start
            if seq.is_prefill():
                prefill_tokens += n
            else:
                decode_seqs += 1
            context += before + n
            token_ids.extend(self.tokens[seq_id][before:before + n])
            slices.append(SeqSlice(start, n, self.scheduler.tables[seq_id].block_ids, before + n))
            if n == seq.num_uncomputed():
                sample_rows[seq_id] = start + n - 1
            start += n

        logits = self.runner.forward(token_ids, slices)
        # One argmax over the sampled rows and one host copy, instead of a sync per sequence.
        sampled: dict[int, int] = {}
        if sample_rows:
            rows = torch.tensor(list(sample_rows.values()), dtype=torch.long, device=logits.device)
            picks = logits.index_select(0, rows).argmax(dim=-1).tolist()
            sampled = dict(zip(sample_rows, picks))
        for seq_id, token in sampled.items():
            self.tokens[seq_id].append(token)
        finished = self.scheduler.update(out, sampled)

        # The step takes the cost model's time (P11, milliseconds). Tokens and finishes are stamped at its end.
        self.clock.charge(step_latency(prefill_tokens, decode_seqs, context) / 1000.0)
        step_end = self.clock.now()
        for seq_id in sampled:
            self.logs[seq_id].token_times.append(step_end)
        for seq in finished:
            self.logs[seq.seq_id].finish_time = step_end
        in_use = set().union(*(table.block_ids for table in self.scheduler.tables.values()))
        self.peak_in_use_blocks = max(self.peak_in_use_blocks, len(in_use))
        self.peak_occupancy_blocks = max(self.peak_occupancy_blocks, self.block_pool.num_blocks - self.block_pool.num_free)
        return [seq.seq_id for seq in finished]

    def request_logs(self) -> list[RequestLog]:
        """Per-request events in request order, ready for bench.metrics."""
        return [self.logs[seq_id] for seq_id in sorted(self.logs)]

    def run(self, max_steps: int = 100_000) -> dict[int, list[int]]:
        """Step until nothing is waiting or running. Returns each request's output tokens."""
        steps = 0
        while self.has_work():
            if steps >= max_steps:
                raise RuntimeError(f"engine did not finish within {max_steps} steps")
            self.step()
            steps += 1
        return {seq_id: self.tokens[seq_id][self.seqs[seq_id].prompt_len:] for seq_id in self.seqs}
