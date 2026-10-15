# Fix log

One entry per fix. Each names the bug, how it showed up, the root cause, the fix, the test that covers it, and one question to answer.

---

## 1. Oversized request accepted, then half-removed in step()

- **Bug:** `add_request` accepts a request the pool can never hold.
- **How it showed up (or would have):** `step()` raised `ValueError` mid-run after the heap had already popped the request. The request stayed in `seqs` and `logs` but was never scheduled. Found by the re-review; reproduced there.
- **Root cause:** capacity was checked only in `Scheduler.add_request`, which runs at release time, after the state had already changed.
- **Fix:** `Engine.add_request` checks the prompt, `max_new_tokens` and pool capacity before it touches any state (`engine.py:Engine.add_request`).
- **Test:** `tests/test_review_fixes.py::test_oversized_request_is_rejected_before_any_state_changes`, plus `test_empty_prompt_and_zero_output_are_refused`.
- **Question:** why must the checks run before the heap push, and not after it?

## 2. Request with a past arrival not visible until the next step

- **Bug:** a request added mid-run with an arrival time already in the past waited for the next `step()` before the scheduler saw it.
- **How it showed up (or would have):** one step of latency for a request that was already due. Found by the re-review.
- **Root cause:** `add_request` always went through the pending heap, and only `step()` released from it.
- **Fix:** `Engine.add_request` calls `_release_due()` after the push, so due requests reach the scheduler immediately. The idle clock jump stays in `_release_arrived`, which runs at the start of `step()` (`engine.py:Engine._release_due`).
- **Test:** `tests/test_review_fixes.py::test_past_arrival_added_mid_run_is_visible_to_the_scheduler_at_once`.
- **Question:** why does releasing a due request in `add_request` not also move the clock?

## 3. Invariant checker missed two kinds of corruption

- **Bug:** the checker did not catch pending hashes that belong to a sequence that is not running, or hash maps that are not exact inverses.
- **How it showed up (or would have):** a stale entry would pass every check and only show up later as a wrong hit. Found by the re-review.
- **Root cause:** `check` verified block owners and the LRU, but nothing about the hash maps or the pending list.
- **Fix:** `invariants.check` now verifies that the two hash maps are exact inverses, and that every pending entry belongs to a running sequence, is held by that sequence, and is not yet registered.
- **Test:** `tests/test_review_fixes.py::test_checker_rejects_pending_hashes_of_a_sequence_that_is_not_running` and `::test_checker_rejects_hash_maps_that_are_not_inverses`.
- **Question:** which of the new checks would catch a block whose hash is registered twice?

## 4. Failed `adopt` left the cache holding a sequence that is not running

- **Bug:** if `BlockTable.adopt` (or the token update) failed after `PrefixCache.admit` succeeded, the cache kept the blocks for a sequence that never started. Those blocks leak.
- **How it showed up (or would have):** a leak that the invariant checker would report only after the exception had been swallowed somewhere above. Found by the re-review.
- **Root cause:** `_admit_cached` made two changes, the cache admit and the table adopt, with no rollback between them.
- **Fix:** `Scheduler._admit_cached` wraps the adopt and the token update. On an exception it frees the cache entry, forgets the table, and re-raises (`scheduler.py:Scheduler._admit_cached`).
- **Test:** `tests/test_review_fixes.py::test_failed_adopt_rolls_back_the_cache_admit`. It restores a cache hit first, so the rollback has real blocks and LRU state to undo.
- **Question:** why is calling `prefix_cache.free` safe on blocks that were admitted a moment ago?

## 5. NaN or infinite arrival time corrupted the heap order

- **Bug:** `add_request` accepted NaN, inf and -inf arrival times. NaN compares false against everything, so the heap order silently breaks.
- **How it showed up (or would have):** requests released in the wrong order, or never, with no error. Found by the re-review.
- **Root cause:** no validation of the arrival time.
- **Fix:** `Engine.add_request` raises `ValueError` for any non-finite arrival, before any state changes (`engine.py:Engine.add_request`).
- **Test:** `tests/test_review_fixes.py::test_non_finite_arrival_time_is_refused`, parametrized over NaN, inf and -inf.
- **Question:** why does a NaN in a heap break the order silently rather than raising?

## 6. Victim choice depended on list order when arrivals tied

- **Bug:** `choose_victim` picked the newest by arrival time only, so among equal arrivals the result depended on the order of the running list.
- **How it showed up (or would have):** the same workload could preempt different sequences depending on how the list happened to be ordered. Found by the re-review.
- **Root cause:** `max` with a key that is only the arrival time returns the first maximal element it meets.
- **Fix:** `choose_victim` keys on `(arrival_time, seq_id)`, so the larger id is the newer one on a tie (`scheduler.py:choose_victim`).
- **Test:** `tests/test_review_fixes.py::test_choose_victim_breaks_arrival_ties_on_seq_id`.
- **Question:** does the new key change the victim when the arrival times differ?

---

## 7. Sweep points could finish short without anyone noticing

- **Bug:** a sweep point could run with requests that never finished, and the baselines and the engine could see different request sets.
- **How it showed up (or would have):** systems compared on different request sets, and no record of a short run. Would have shown up as a throughput number that quietly excluded work.
- **Root cause:** `run_sweep` did not drop requests that cannot fit, and did not check that every submitted request finished.
- **Fix:** `run_sweep` drops oversized requests for every system and records the count. Each point asserts that finished equals submitted. Each record carries `submitted`, `finished`, `rejected` and `oversized`, and `validate_result` enforces the rules (`bench/sweep.py:run_sweep`, `bench/results.py:validate_result`).
- **Test:** `tests/test_sweep_integrity.py::test_oversized_requests_are_dropped_for_every_system_and_counted`, and the rule tests in `tests/test_results_schema.py`.
- **Question:** why must the oversized filter apply to the baselines too, even though they could run those requests?

## 8. KV utilization was not recorded

- **Bug:** the sweep recorded no KV utilization, so the two numbers that matter for paging, blocks in use and pool occupancy, were unavailable.
- **How it showed up (or would have):** the notebook could report throughput but not how much of the pool was in use.
- **Root cause:** the engine kept no peaks.
- **Fix:** the engine tracks peak blocks held by running sequences (in use) and peak non-free blocks (occupancy, which includes the retained LRU). Both are recorded per point and labelled. Plots use in use (`engine.py:Engine.step`, `bench/sweep.py:run_sweep`, `bench/plots.py`).
- **Test:** `tests/test_results_schema.py::test_in_use_cannot_exceed_occupancy_or_the_pool` and `::test_engine_record_must_report_kv_numbers`.
- **Question:** why can in use never exceed occupancy?

---

## 9. Prefix caching diagnostic: no bug in the forwarding path

- **Bug:** none found. The check was whether caching cut the prompt tokens sent through the model by about the cached count.
- **How it showed up (or would have):** it would have shown up as forwarded prompt tokens that stayed flat with caching on. They did not. On the cost-model clock, prompt tokens forwarded fell from 2,043 to 571, a drop of exactly the 1,472 cached tokens.
- **Root cause:** not applicable. The count is taken after admission credits cache hits, so cached tokens are never forwarded.
- **Fix:** none needed. The diagnostic is `scripts/prefix_diagnostics.py` (run on a GPU; it writes `prefix_diag_results.json`).
- **Test:** none. The forwarded-token count is measured only by the diagnostic script; the shared-prefix sweep test checks the hit rate, not the forwarded tokens.
- **Question:** why does the wall-clock comparison give different tokens with caching on and off, when the cost-model comparison does not?

**Numbers** (3050, fp16, 24 requests, shared prefix 64, rate 6, max batch 8, 2,043 prompt tokens in total):

| | caching off | caching on |
|---|---|---|
| prompt tokens forwarded | 2,043 | 571 |
| cached tokens | 0 | 1,472 |
| steps (forward calls) | 317 | 321 |
| TTFT p50 / p99, cost-model clock | 9.7 / 18.6 ms | 6.7 / 11.8 ms |
| TPOT p50, cost-model clock | 5.4 ms | 5.4 ms |
| wall time, cost-model clock | 68.8 s | 48.5 s |
| identical tokens, cost-model clock | yes | |
| TTFT p50 / p99, wall clock | 0.38 / 4.13 s | 0.32 / 4.37 s |
| TPOT p50, wall clock | 186 ms | 192 ms |
| wall time, wall clock | 12.8 s | 13.1 s |
| identical tokens, wall clock | no (see question) | |

**Explanation of wall time.** On the cost-model clock, wall time fell 30%, with forward calls about equal. Per forward call, the time went from 0.22 s to 0.15 s, because the prefill chunks are smaller. On the wall clock, the run is only 13 s and the difference is within noise. Two wall-clock runs with caching off already differ, at 82 and 85 steps, and an earlier identical setting measured 117 s and 120 s. At this scale, decode steps and the fixed per-forward cost set the step count. Caching removes prefill tokens, so it cuts time per forward, not the number of forwards. The clear gain is TTFT on the cost-model clock. On the wall clock the p50 improves (0.38 to 0.32 s), but the p99 does not (4.13 to 4.37 s). With 24 requests, p99 is essentially the slowest request, which here is set by arrival bursts, not by the prefix.

**Why the wall-clock tokens differ.** With the wall clock, arrivals and step timing decide which requests share a batch. Batch shape changes fp16 numerics, so two runs with caching off already differ. The cost-model clock fixes the schedule, and there the outputs are identical with caching on and off. The wall-clock token comparison is therefore not a valid test of caching correctness.

---

## 10. Every layer copied its slot list to the GPU

- **Bug:** a decode step on the 3050 made 851 synchronizing calls, 768 of them from one line: the slot list for each sequence was copied to the GPU in every one of the 24 layers.
- **How it showed up:** the step took about 530 ms at batch 32 with the GPU 76% idle in the profiled run. Step time grew linearly with batch size.
- **Root cause:** the slot list depends only on the block table and context length, which do not change within a step. `paged_backend_forward` rebuilt it inside the layer loop.
- **Fix:** `RealRunner.forward` builds input ids, positions, write slots and the context slot index into one host list, and copies it to the device once per step. `StepContext.ctx_index` carries the context index; `paged_backend_forward` takes a slice per sequence. When `ctx_index` is None (direct callers), the backend builds it once per call (`minivllm/attention/backend.py:paged_backend_forward`, `minivllm/runner/real.py:RealRunner.forward`).
- **Test:** `tests/test_step_overhead.py::test_decode_step_syncs_do_not_grow_with_batch_size` (CUDA). It counts sync-debug warnings in one decode step: 3 at batch 2 and 3 at batch 8, against 851 before.
- **Question:** why does the per-step sync count no longer depend on the number of layers, and which three syncs remain?

## 11. KV pool write checked its slot range on the device, twice per layer

- **Bug:** `KVPool.write` ran two `torch.any` range checks on device tensors. That is 2 syncs per layer, 48 per step.
- **How it showed up:** the same 851-sync count; removing the checks alone moved step time by less than the noise, because the copy in fix 10 was the larger stall.
- **Root cause:** a device-side `any()` returns a value the host must read before it can continue.
- **Fix:** `KVPool.write(..., check_range=True)`. The runner has already checked the slots on the host (they come from block tables), so the backend passes `check_range=not context.validated`. The default still checks, so direct callers keep the P19 errors (`minivllm/attention/kv_pool.py:KVPool.write`).
- **Test:** `tests/test_step_overhead.py::test_write_without_range_check_stores_the_same_values` and `::test_default_write_still_rejects_out_of_range_slot`.
- **Question:** why keep the check as the default when the hot path turns it off?

## 12. Sampling read one token per sequence from the device

- **Bug:** `Engine.step` called `int(logits[row].argmax())` for each sequence, so 32 syncs per step at batch 32.
- **How it showed up:** the sampling and bookkeeping phase took 31 ms per step (4.8%) in the profile.
- **Root cause:** each `int()` reads one device scalar and blocks the host until the GPU reaches it.
- **Fix:** gather the sampled rows into one index tensor, take one `argmax(dim=-1)` over them, and copy the result with one `.tolist()` (`minivllm/engine.py:Engine.step`).
- **Test:** the greedy-equals-HF gates (`tests/test_p23_backend.py::test_real_model_greedy_matches_generate` with `MINIVLLM_REAL_MODEL`, and `tests/test_baselines.py` gate tests), plus the sync count test above.
- **Question:** the step still copies the row index to the device. Why is that one copy not a per-sequence sync?

## 13. Decode attention looped over sequences in every layer

- **Bug:** `paged_backend_forward` ran one gather and one attention call per sequence per layer. A decode step at batch 32 made 768 such calls, about 10 kernel launches each.
- **How it showed up:** an isolated backend call for one layer took 17 ms at batch 32, and the same time whether or not the GPU was waited on. The step time scaled linearly with batch.
- **Root cause:** the attention from P20 and P23 handles one sequence per call. The host spends about 0.6 ms per sequence-layer enqueuing kernels, so the call is launch-bound.
- **Fix:** `paged_decode_attention` (`minivllm/attention/batched.py`) does one gather over a padded block table and one masked attention call for all decode rows. `RealRunner.forward` builds the decode plan once per step, and `paged_backend_forward` routes length-1 sequences through the batched call and prefill chunks through the old path. `batched_decode=False` on `RealRunner` or `Engine` restores the old path for A/B tests.
- **Test:** `tests/test_p23b_batched.py` (agreement with the reference, NaN padding, flat kernel count) and `tests/test_p23b_wiring.py` (identical logits and identical engine outputs with and without the flag).
- **Question:** why is the kernel count the same for batch sizes 2 to 64 but one higher at batch 1?

## 14. Padding would leak NaN through a masked softmax

- **Bug:** found in the design, not in a run. Masking only the scores would leave `0 * NaN = NaN` in the probability-times-V matmul whenever a padded or unused slot holds NaN.
- **Root cause:** a softmax weight of exactly 0 does not zero a NaN value; the product is NaN.
- **Fix:** positions at or past `context_len` get score -inf and their V is set to 0 before the matmul (`minivllm/attention/batched.py:paged_decode_attention`). I did not use a reserved null block as vLLM does: it would take one block id out of `BlockPool` (changing P1 accounting) and would still depend on that block staying zero. Masking V keeps the pool and the allocator unchanged.
- **Test:** `tests/test_p23b_batched.py::test_nan_in_every_unreferenced_slot_leaves_the_output_unchanged` fills every slot no real position uses with NaN and requires finite, unchanged output.
- **Question:** if the V mask were removed, which slot of the test would produce the NaN, and in which matmul?

## 15. Prefill computed logits for every prompt row

- **Bug:** `RealRunner.forward` ran `lm_head` on every packed row, so a prefill of L prompt tokens built an [L, vocab] logits tensor (151,936 columns for Qwen2.5-0.5B). The engine then kept one row per sequence. For a 256-token prompt that tensor is about 78 MB in fp16, and 255 of its rows are never read.
- **How it showed up:** found by reading `RealRunner.forward` for the TTFT work. The sampled rows were picked out of the full logits with `index_select`.
- **Root cause:** `forward` always returned `outputs.logits[0]` for the whole packed row, because the model was called without `logits_to_keep`.
- **Fix:** `RealRunner.forward(..., sample_rows=...)` passes the sampled packed rows to the model as a tensor `logits_to_keep`. transformers' Qwen2 forward selects those hidden rows before `lm_head`, so only those rows are projected. The rows are sent in the step's one host copy, as the other per-step tensors are. `sample_rows=None` keeps the full [T, vocab] result for the callers that need every row (`gpu_checks.teacher_forced_agreement`, the P23 tests). `Engine.step` passes its sample rows and takes an argmax over the returned rows (`minivllm/runner/real.py:RealRunner.forward`, `minivllm/engine.py:Engine.step`).
- **Test:** `tests/test_sample_rows.py` (the picked rows equal the full rows, out of order, on a prefill and on a batched decode step; an empty list still writes the KV). The full suite gives 369 passed and 3 skipped before and after the change. On the 3050, `scripts/gpu_checks_3050.py` gives the same agreement numbers, and the identical-outputs gate passes (`results/3050/`).
- **Timing:** `scripts/prefill_timing.py`, run from a `git worktree` at `fe82a1e` (before) and from the working tree (after), in four alternating serial sessions on the 3050. Each value is the median of 7 repeats within a session; the table gives the median of the four session medians. Raw numbers: `results/3050/prefill_{before,after}_s{1..4}.json`.

| Requests x prompt length | Before, ms (sessions) | After, ms (sessions) | After / before |
|---|---|---|---|
| 1 x 64 | 130.2 (132, 129, 164, 124) | 137.6 (134, 273, 142, 119) | 1.06 |
| 1 x 256 | 137.8 (129, 135, 172, 140) | 132.1 (126, 260, 138, 119) | 0.96 |
| 8 x 64 | 352.4 (350, 355, 441, 336) | 341.5 (329, 720, 353, 322) | 0.97 |
| 8 x 256 | 366.9 (365, 368, 427, 359) | 337.2 (328, 687, 346, 318) | 0.92 |

- The second "after" session ran about twice as slow as every other session, before and after alike, with the GPU idle and no other compute. The laptop's clock was at 210 MHz at the time, so the medians are the more reliable figure, and even they sit inside the session spread.
- TTFT here is the prefill step time, because each request has arrival time 0 and a one-token output, so its first token comes out of that one step.
- The first generated tokens match between the two code paths in all four sessions.
- **Reading:** the change removes the vocabulary projection for the prompt rows, but the gain is inside the noise at these sizes, and the largest median gain is 8% (8 x 256). The prefill step on this card is dominated by the per-layer work, not by `lm_head`. The memory saving is the arithmetic above (about 78 MB per 256-token prompt) and was not measured here.
- **Question:** the 8 x 256 prefill still builds the full [1, 2048, 896] hidden state for every layer. Which of its tensors would be the next to shrink, and what would it take to avoid building it?

---

## Open items (not fixes)

- **Fast-suite flake: not reproduced.** The fast suite (`pytest -m "not slow"`) ran 20 times in file order and 5 times in seeded random orders (`random.Random(seed).shuffle` over the collected node IDs; `pytest-randomly` is not installed). All 25 runs gave 366 passed and 3 skipped, at commit 83ccae9. No failing test was named, so no root cause or fix is recorded. One run overlapped with a second pytest process on the same GPU and still passed. The tests most sensitive to timing or to CUDA asynchrony are `tests/test_wall_clock.py`, `tests/test_engine_arrivals.py`, `tests/test_step_overhead.py` (sync counts) and `tests/test_p23b_batched.py::test_kernel_launches_do_not_grow_with_batch_size` (profiler trace counts). The log of the failure is needed to test them.
- **Prefix caching does not give identical tokens in fp16 on the 3050.** `scripts/prefix_diagnostics.py` reports `identical_tokens: false` with the FakeClock and with the WallClock run on `fe82a1e` as well. With the current code the WallClock run is True, so the WallClock result varies between runs. The forwarded-token counts are exact (2,043 to 571). The likely cause is that prefix caching changes which requests share a batch, and fp16 arithmetic depends on batch composition. That is not isolated here. The gate that proves exactness uses float64 (`results/3050/identical_outputs_gate.json`).
