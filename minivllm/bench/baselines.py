"""Hugging Face baselines: sequential and static batching, on the same clock and cost model as the engine.

Plumbing, not a numbered problem.

Operations
    run_sequential(model, jobs, clock) -> (outputs, logs)
        one request at a time, in arrival order
    run_static(model, jobs, clock, max_batch) -> (outputs, logs)
        groups of max_batch requests in arrival order. A batch starts once its last
        request has arrived, pads prompts on the left, and runs until its longest
        output is done. A request's finish_time is the end of its batch.
    outputs: dict[request_id, list[int]], exactly output_len tokens per request

Timing
    Every forward pass inside generate() is charged to the clock with the same cost
    model as the engine (minivllm.runner.fake.step_latency, in milliseconds):
        prefill step   step_latency(B * P, 0, B * P)       B rows, P padded prompt length
        decode step    step_latency(0, B, B * current_len)
    With B = 1 this is exactly the engine's cost for one request, so a single
    request gets the same token times from either system.

Ignore EOS
    min_new_tokens = output_len suppresses end-of-sequence until the requested
    length, so every system emits the same number of tokens.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from minivllm.bench.jobs import Job
from minivllm.bench.metrics import RequestLog
from minivllm.clock import Clock
from minivllm.runner.fake import step_latency


class _StepTimer(LogitsProcessor):
    """Charges each generate() step to the clock and records when its tokens came out."""

    def __init__(self, clock: Clock, batch_size: int, padded_prompt_len: int) -> None:
        self.clock = clock
        self.batch_size = batch_size
        self.padded_prompt_len = padded_prompt_len
        self.stamps: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.stamps:
            tokens = self.batch_size * self.padded_prompt_len
            ms = step_latency(tokens, 0, tokens)
        else:
            ms = step_latency(0, self.batch_size, self.batch_size * input_ids.shape[1])
        self.clock.charge(ms / 1000.0)
        self.stamps.append(self.clock.now())
        return scores


def _generate(model, input_ids, attention_mask, max_new: int, timer: _StepTimer) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            # Greedy means plain argmax. Instruct checkpoints ship repetition_penalty in generation_config,
            # which generate() applies even when do_sample is False, so turn it off explicitly.
            repetition_penalty=1.0,
            max_new_tokens=max_new,
            min_new_tokens=max_new,
            eos_token_id=None,
            pad_token_id=0,
            logits_processor=LogitsProcessorList([timer]),
        )
    return out[:, input_ids.shape[1]:]


def run_sequential(model, jobs: list[Job], clock: Clock) -> tuple[dict[int, list[int]], list[RequestLog]]:
    outputs: dict[int, list[int]] = {}
    logs: list[RequestLog] = []
    for job in sorted(jobs, key=lambda j: j.arrival_time):
        clock.advance_to(job.arrival_time)
        dequeue = clock.now()
        ids = torch.tensor([job.prompt_ids], dtype=torch.long, device=model.device)
        timer = _StepTimer(clock, batch_size=1, padded_prompt_len=len(job.prompt_ids))
        tokens = _generate(model, ids, torch.ones_like(ids), job.output_len, timer)
        outputs[job.request_id] = tokens[0].tolist()
        logs.append(RequestLog(job.request_id, job.arrival_time, dequeue, timer.stamps, timer.stamps[-1]))
    return outputs, logs


def run_static(model, jobs: list[Job], clock: Clock, max_batch: int) -> tuple[dict[int, list[int]], list[RequestLog]]:
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    ordered = sorted(jobs, key=lambda j: j.arrival_time)
    outputs: dict[int, list[int]] = {}
    logs: list[RequestLog] = []
    for first in range(0, len(ordered), max_batch):
        batch = ordered[first:first + max_batch]
        clock.advance_to(max(job.arrival_time for job in batch))
        dequeue = clock.now()
        padded = max(len(job.prompt_ids) for job in batch)
        ids = torch.zeros(len(batch), padded, dtype=torch.long, device=model.device)
        mask = torch.zeros(len(batch), padded, dtype=torch.long, device=model.device)
        for row, job in enumerate(batch):
            ids[row, padded - len(job.prompt_ids):] = torch.tensor(job.prompt_ids, dtype=torch.long, device=model.device)
            mask[row, padded - len(job.prompt_ids):] = 1
        timer = _StepTimer(clock, batch_size=len(batch), padded_prompt_len=padded)
        tokens = _generate(model, ids, mask, max(job.output_len for job in batch), timer)
        batch_end = timer.stamps[-1]
        for row, job in enumerate(batch):
            outputs[job.request_id] = tokens[row, :job.output_len].tolist()
            logs.append(
                RequestLog(job.request_id, job.arrival_time, dequeue, timer.stamps[:job.output_len], batch_end)
            )
    return outputs, logs
