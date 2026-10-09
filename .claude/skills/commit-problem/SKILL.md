---
name: commit-problem
description: Commit the work for problem N at the end of a session — the solution file, its tests, and the progress/design-log updates — as one git commit. Use only when the human says "/commit-problem N" or "commit problem N".
---

# Commit problem N

Makes one commit containing the work for problem N. It only runs when the
human asks. It never pushes.

## Procedure

1. Check the repo. Run `git rev-parse --is-inside-work-tree`. If it is not a
   git repo, say so and stop. Ask the human before running `git init`.
2. Read `docs/progress.md`. Problem N should be "tests pass" or later. If it
   is still "todo" or "scaffolded", say so and stop. Committing unfinished
   work is the human's call, not this skill's.
3. Run the tests for problem N with `python -m pytest tests/test_pNN_<name>.py -q`.
   If any fail, stop and report the failing names. Do not commit red code.
4. Run `git status --short`. Show the list to the human.
5. Stage only the problem's files:
   - the solution file(s) named in the phase doc for problem N
   - `tests/test_pNN_<name>.py`
   - `docs/progress.md`
   - `docs/design-log.md`, if it changed
   Do not stage `__pycache__`, `.pytest_cache`, scratch files, or anything
   outside this list. If the repo has no `.gitignore` covering those caches,
   say so and leave them unstaged.
6. Commit with the message:

   ```
   P{N}: <problem name>

   <one or two sentences on what the solution does>

   Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>
   ```

   Use the problem name from `docs/progress.md`. Pass the message with a
   heredoc so the trailer stays intact.
7. Run `git log -1 --stat` and show the result.

## Limits

- Never `git push`, `git commit --amend`, `git reset`, or `--no-verify`.
- If a pre-commit hook fails, report it and stop. Do not bypass it.
- One problem per commit. If other unrelated changes are staged, unstage
  them and say so.
