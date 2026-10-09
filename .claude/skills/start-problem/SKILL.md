---
name: start-problem
description: Scaffold problem N of the mini-vLLM project — stub, failing tests, fixtures — without writing the solution. Use when the human says "/start-problem N" or "start problem N".
---

# Start problem N

1. Read `docs/progress.md`, then only the phase doc containing problem N.
   Do not open other phase docs.
2. Check prerequisites: earlier problems this one depends on should be at
   "tests pass" or later. If not, say which, and stop.
3. Create the stub in the file named by the spec: exact signature, type hints,
   body `raise NotImplementedError`. Existing human-written code in that file
   stays untouched.
4. Write the problem statement as a DSA-style docstring on the stub (see
   "Problem statement format" below). The human must be able to understand
   what to build from the docstring alone, without opening the phase doc.
5. Write `tests/test_pNN_<name>.py` covering every "Tests to write" item plus
   any edge case you think the spec missed (mark those `# extra edge case`).
   Tests must fail against the stub. Run them to confirm.
6. Write any non-problem scaffolding the problem needs (fixtures, harness,
   runner shells). That code you may write fully.
7. Set problem N to "scaffolded" in `docs/progress.md`.
8. Reply with: the file to edit, the test command, and the checkpoint question.
   Nothing else — no hints about the solution unless asked.

Never write any part of the solution body. If the human asks, follow the hint
ladder in `AGENTS.md`.

## Problem statement format

Put this at the top of the module or class docstring that the human implements.
Write it like a LeetCode / DSA problem, so the human knows the contract before
they touch the code:

```
"""
<Name> — Problem N

Story (2–4 sentences): what real-world situation this models and why the
engine needs it. No implementation hints.

Operations
    method(args) -> return     one line on what it does
    ...                        one entry per public method

Errors
    Which call raises which exception, and when. Name the exception class.
    Say explicitly if the spec leaves a choice open, and list the options.

Example
    pool = BlockPool(3)
    pool.num_free          -> 3
    a = pool.allocate()    -> one of {0, 1, 2}, and it is not returned again
                              until freed
    pool.num_free          -> 2
    ...
    Use a short trace with concrete values. Show state after each step.

Constraints
    Complexity targets, value ranges, block size, anything the spec demands.

Edge cases to think about
    Bullet list of cases the tests check (empty, full, boundary, repeated
    operations). Phrase as questions where the answer is the human's to find.

Tests
    Name the test file and what it covers. Do not restate test bodies.
"""
```

Rules for the statement:
- Every example output must be correct for the spec. Re-check the trace
  against the tests before replying.
- Do not describe data structures or algorithms that would reveal the
  solution (e.g. "use a free list" or "use a stack"). State the required
  behavior and complexity only.
- If a spec detail is ambiguous, pick one, state it in the docstring, and
  mention the choice in the reply so the human can redirect it.
