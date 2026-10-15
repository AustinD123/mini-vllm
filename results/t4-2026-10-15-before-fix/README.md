# T4 baseline, before the P23b and sync fixes

This run predates the P23b and sync fixes. It was made at commit `cd06f22`
("Colab: three repeats per wall-clock point, simulated preemption sweep, runtime estimate").

Commits after `cd06f22` that change the numbers here:

- `aba809b`: engine step makes one host-to-device copy per step, and samples in one batch.
- `05cec45` and `4d809fd`: P23b batched decode attention, wired into the backend and runner.

Compare against a T4 run from a later commit before reading a speedup into it. The
per-step sync and batched-attention changes alter decode time directly.

Contents: the owner's pre-fix Colab results, copied here by the owner. The zip itself is gitignored (`*.zip`).
