---
name: beautify-problem
description: Clean up the human's finished solution for problem N so it matches the repo's layout and conventions — formatting, imports, type hints, docstrings, dead code — without changing behavior. Use only when the human says "/beautify-problem N" or "beautify problem N" after their tests pass.
---

# Beautify problem N

Runs after the human's implementation of problem N passes its tests. It makes
the code match the repo's conventions. It does not change what the code does
and it does not change the algorithm.

This skill edits the human's code, which AGENTS.md otherwise forbids unless
asked. Running `/beautify-problem N` is that ask. Stay inside the limits below.

## Preconditions

1. Read `docs/progress.md`. Problem N must be at "tests pass" or later.
   If it is not, say so and stop.
2. Run `python -m pytest tests/test_pNN_<name>.py -q`. Record the result.
   If anything fails, stop. Beautifying broken code hides the bug. Tell the
   human to run `/review-problem N` first.
3. Save a copy of each file you will touch, under the scratchpad directory,
   so you can revert if a check fails.

## What you may change

- Formatting: indentation, line length, blank lines, quote style, trailing
  whitespace. Match the rest of the repo.
- Imports: order them, remove unused ones, use the repo's import style.
- Names: rename a local or a private helper if it is unclear. Do not rename
  public methods, classes, or anything a test or the phase spec references.
- Type hints: add missing hints on signatures and fields.
- Docstrings: make them match the repo style. The problem statement in the
  module docstring stays. Remove comments that narrate the code line by line.
- Dead code: remove unused variables, unreachable branches, debug prints,
  commented-out code.
- Repo layout: if the code sits in the wrong file or module relative to the
  AGENTS.md layout, tell the human where it should go. Move it only if the
  human agrees. Do not move it on your own.

## What you must not change

- Any behavior: return values, exceptions, side effects, ordering of
  operations, complexity. If you are unsure whether a change alters behavior,
  do not make it.
- The algorithm or data structure the human chose. Do not swap in a "better"
  one.
- Public signatures named in the spec.
- Tests. Do not edit `tests/` to make them pass.
- Anything outside the problem's file(s), except the imports the move needs.

If a cleanup would be a real logic change, list it as a suggestion and leave
the code alone. The human decides.

## Procedure

1. Read the file(s) the spec names. Read only the phase doc for problem N if
   you need the file list.
2. Make the allowed changes, one kind at a time. Keep edits small.
3. Re-run the same test command from the preconditions. The results must
   match the recorded result exactly. If any test changes status, revert the
   file from your saved copy and report which change caused it.
4. If the repo has a formatter configured (check for `pyproject.toml`,
   `ruff.toml`, `setup.cfg`), run it on the touched files. Do not
   introduce a formatter the repo does not already use.
5. Do not change `docs/progress.md` status. Beautifying does not move a
   problem forward. `/review-problem N` does that.

## Reply

Keep it short:

- Files changed, and one line per kind of change (e.g. "formatting, unused
  import removed, type hints added to `allocate`").
- Test result before and after. Both must be identical.
- Suggestions you did not apply, each with a one-line reason. Put logic
  concerns here.
- If the layout is off, say where the code should live.
- No solution hints. No rewritten logic.
