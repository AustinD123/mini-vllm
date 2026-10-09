---
name: review-problem
description: Review the human's solution to mini-vLLM problem N, quiz them on the checkpoint, compare with vLLM, and log learnings. Use when the human says "/review-problem N" or "review problem N".
---

# Review problem N

1. Run the tests for problem N. If they fail, report which and why in one
   line each, and stop. Do not fix them.
2. Review the human's code for: correctness bugs the tests miss, unhandled
   edge cases, complexity (does it meet the spec's constraints?), invariants
   that could break later problems. Describe each issue and where it is; do
   not write the fix. If you can, add a test that exposes it.
   If subagents are available, do this review in a fresh subagent that hasn't
   seen the scaffolding, so the reviewer isn't grading its own tests.
3. Quiz: ask the checkpoint question from the phase doc, plus one follow-up
   you choose. Wait for the answer. Correct misconceptions; don't accept a
   vague answer.
4. vLLM comparison: if the spec lists a vLLM ref, open that file in the
   current vLLM repo (clone sparse if needed: `vllm/v1/core`) and point the
   human to the 1–2 functions that correspond. Summarize the main difference
   in two or three sentences. Don't paste large chunks.
5. Ask the human to fill the design-log entry in `docs/design-log.md`; offer
   the template prefilled with headings only.
6. Once the checkpoint is answered well, set problem N to "reviewed" in
   `docs/progress.md` and advance "Current problem".
