# Unified tasks and cooperative stop — engineering verification

Date: 2026-09-27. Checkout: main at `80a28de`, with uncommitted implementation.
This record does not authorize a research run, commit, release or platform claim.

## Completed checks

- Base suite: **1655 passed, 265 skipped**, exit 0. Optional dependencies were
  intentionally absent from `/private/tmp/smartsom-base-checks`.
- Optional CPU suite: **39 passed**, exit 0, in the existing
  `/private/tmp/smartsom-tuning-env`. It covered native stop tests, composable
  workflow, Tune scheduler/callback and batch regressions.
- Focused base entry/control suite: **111 passed, 6 skipped**, exit 0.
- Ruff lint and formatting: passed; 432 Python files already formatted.
- `uv lock --check`: passed, 121 packages; `git diff --check`: passed.
- Current main environment's `smartsom stop --help` and rule-only
  `smartsom check --task evaluate` completed successfully without execution.

All test outputs used independent `/private/tmp/smartsom-unified-*` directories.
The base environment ran with an absolute `PYTHONPATH` so subprocess checks from
other working directories imported the current checkout. Earlier development
checks exposed old directory-layout expectations and that missing import path;
those were corrected. Editing source while a Study preparation test was running
also correctly invalidated its frozen implementation identity. The final passing
suite ran after source edits stopped; strict identity checks were not weakened.

## Observed behavior

- Task checks are read-only; fresh model evaluation, incompatible task/input
  combinations, preview conflicts and checkpoint conditions are rejected.
- Preview uses one case and one repetition, retains selected model provenance,
  marks its output, and leaves its source files unchanged.
- Real PPO and DQN CPU updates stopped after complete recovery saving, then
  resumed to their original budget. Source checking and preview used an actual
  saved `last` checkpoint. An absent `best` was correctly rejected; short
  truncated validation is not assumed to create a best model.
- A spawned parallel Study stopped dispatch, drained active workers and left
  pending entries queued with the Study interrupted.
- Tune calibration cancelled its owned probe and retained the calibration
  report without launching formal attempts.
- A real local Ray Tune PPO trial stopped at saved progress (8 physical ticks),
  then fixed execution resumed to the original 64-tick budget. The final batch
  ledger recorded completion. This was a tiny engineering case, not a resource
  performance benchmark or a policy-quality result.
- Ordinary timeout left the test driver alive. Explicit force terminated only
  the test driver and registered child; an unrelated test process stayed alive.
  PID reuse, old request generations and unregistered legacy runs were tested.
- Rendering and recording flags reached v3 execution through a stubbed live
  window. This establishes parameter forwarding, not visual UI acceptance.

## Boundaries

macOS CPU is the executed platform. Linux, Windows, CUDA/H20, Slurm/CARC,
multi-node execution and actual graphical interaction were not qualified here.
Stop does not add background persistence or full single-run evaluation recovery.
Strict source/implementation checks still apply to resume. Existing unregistered
running experiments were not modified, stopped or retrofitted.

Main's commit stayed unchanged. `configs/factories/large.yaml` retained SHA-256
`ee68146af3d91d065fa2e4537147e364af3c5243fb565d6a87be275816ff48f5`.
No configuration, Studio template or other worktree was edited.
