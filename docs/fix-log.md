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
