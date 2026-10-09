# Phase 2 — Scheduler (problems 6–13)

Runs entirely on `FakeRunner`. Agent scaffolds: `SchedulerOutput` dataclass,
`FakeRunner` shell, the simulation loop driver, workload generator.

Workload generator (agent writes): Poisson arrivals at rate λ; prompt and
output lengths lognormal, clipped to [1, max_model_len]; fixed seed. Parameters
should be fitted from a public trace (e.g. ShareGPT length statistics) and
documented in the README.

---

## P6 — Static batching baseline
**File:** `minivllm/core/static_scheduler.py`
Collect up to B waiting requests, run them together until *all* finish, then
take the next batch. Same `step()` interface as the real scheduler.
**Tests to write:** a batch with output lengths [10, 200] occupies 200 steps;
no new admission while a batch is running.
**Checkpoint:** Who waits for whom, and how bad does it get when output
lengths are lognormal?

---

## P7 — FCFS continuous batching step
**File:** `minivllm/core/scheduler.py`
```python
class Scheduler:
    def __init__(self, pool: BlockPool, block_size: int, max_batch: int, token_budget: int): ...
    def add_request(self, seq: Sequence) -> None: ...
    def step(self) -> SchedulerOutput: ...
    def update(self, output: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]: ...  # returns finished
```
Admit waiting requests in arrival order while blocks and batch slots allow.
Finished sequences free their blocks in `update`.
**Tests to write:** request arriving mid-run joins next step; strict FCFS (a
small request behind a big one doesn't jump ahead); finished seq's blocks are
free next step.
**Checkpoint:** What exactly happens between two steps, and why is that the
whole trick of continuous batching?
**vLLM ref:** `vllm/v1/core/sched/scheduler.py` (`schedule`),
`sched/request_queue.py`.

---

## P8 — Per-step token budget
Extend `step()`: decodes (1 token each) are scheduled first, remaining budget
goes to prefill.
**Tests to write:** budget 64 with 10 decoding seqs leaves 54 for prefill;
budget smaller than number of decodes behaves predictably (document choice).
**Checkpoint:** Why decodes first? What does the budget trade off?

---

## P9 — Chunked prefill
A prompt longer than remaining budget is split across steps.
**Tests to write:** 1000-token prompt, budget 256, no decodes: computed in
4 steps (256, 256, 256, 232); logits/sample only on the last chunk; a decode
running alongside is never starved.
**Checkpoint:** What happens to TPOT of running requests without chunking when
a 4k-token prompt arrives?

---

## P10 — Preemption (recompute)
When a running sequence can't get a block for its next token, preempt the most
recently arrived running sequence: free its blocks, set PREEMPTED, put it at
the *front* of the waiting queue.
**Tests to write:** pool too small for all → preemption occurs and every
request still finishes; victim is the newest; a sequence that can't fit even
in an empty pool is rejected at admission, not looped forever; no admission
of new requests while preempted ones wait.
**Checkpoint:** What happens when the pool runs out mid-sequence, and what does
the scheduler do? How do you rule out livelock?
**vLLM ref:** preemption path in `sched/scheduler.py`.

---

## P11 — Cost model
**File:** `minivllm/runner/fake.py`
```python
def step_latency(num_prefill_tokens: int, num_decode_seqs: int, total_context: int) -> float: ...
```
Linear model with fixed overhead: `a + b*prefill_tokens + c*decode_seqs + d*total_context`.
Coefficients start as guesses, get fitted in P26.
**Tests to write:** monotonic in each argument; zero work → overhead only.
**Checkpoint:** Why is decode memory-bandwidth bound and prefill compute
bound? What does that imply for batching decodes?

---

## P12 — Metrics from an event log
**File:** `minivllm/bench/metrics.py`
Given per-request events (scheduled arrival, first token, each token, finish),
compute TTFT, TPOT, ITL, E2E with p50/p90/p99, throughput, goodput (SLO
TTFT ≤ 2 s, TPOT ≤ 100 ms).
**Tests to write:** hand-built log with known answers; latency measured from
*scheduled* arrival, not from when the request was dequeued.
**Checkpoint:** What is coordinated omission and how would it flatter your p99?

---

## P13 — Invariant checker + fuzz
**File:** `minivllm/core/invariants.py`
`check(scheduler)` asserts after every step: free + owned == total blocks; no
block owned twice; each running seq has enough blocks for its computed tokens;
waiting/running/preempted sets are disjoint.
**Tests to write:** 10k random steps across random workloads and small pools,
invariant checked every step.
**Checkpoint:** Which bug from PageServe's "v3 audit" would this have caught?

**Phase 2 done = blog post part 1** (simulator: paged vs contiguous, continuous
vs static, preemption under pressure).
