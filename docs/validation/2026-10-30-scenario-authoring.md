# Scenario authoring and Studio validation

This supplement changes authoring and editing, not the physical stepping or
learner update rules. Current user inputs are Scenario, Algorithm and Experiment.
See ADR0062 and `docs/scenario-authoring.md` for the contract and migration.

## Local engineering evidence

All checks below used the pinned Python environment on macOS CPU. Test outputs,
Qt screenshots, temporary migrated configs and generated runs stayed outside Git.

- All 25 tracked experiment entries compiled with their new scene references.
- Small/Medium × normal/low/medium/high × seeds 101/203 generated exactly equal
  typed physical worlds before/after conversion. This checks orders, due dates,
  routes, processing work and faults; it is not an algorithm performance claim.
- Defaults and per-machine/per-AGV overrides, separate machine fault streams,
  source freeze/reload, scientific identity, create-only migration and invalid
  size/rotation rejection have focused tests.
- Qt tests cover common settings, explicit overrides/reset, no-op numeric
  precision, partial nested inheritance, undo/redo, save/reopen, recovery and
  deterministic case previews. Machine property editing exposes coordinates
  and one operation; current placement and YAML enforce 1×1 machines/AGVs.
- The full fast selector list executed locally: **1,071 passed, 2 Windows-only
  skips**. This used the development environment with extras, not CI's lean
  Ubuntu environment. Hosted CI remains the lean/platform qualification.
- Additional learning, resource-learning, usability and Gym selector groups:
  **67, 20, 6 and 1 tests passed**, respectively. They cover native updates,
  restore, public execution and tracking. These overlap earlier acceptance and
  must not be added to previous counts as unique tests.
- One existing native evaluation reported incomplete W&B synchronization while
  preserving local evidence. No new online/mobile W&B acceptance is claimed.
  The dedicated tracking group passed its offline/SDK contracts; subsequent
  tests explicitly set WANDB_MODE=disabled.
- Final Studio plus affected configuration/component regression: **201 passed**.
  Earlier failed/intermediate Studio counts are not treated as acceptance.
- Ruff lint/format, whitespace and full CI ownership inventory checks passed.

The initial Studio group exposed a stale test loader: it tried to read the new
scenario file as `smartsom.factory/v2`. It was corrected to verify the actual new
scene envelope, retaining inspection/drawing assertions. Additional review found
that numeric YAML defaults could be unnecessarily pinned during typed conversion;
reconciliation now compares typed resource values. A regression test guards this.

## Boundaries

Offscreen Qt and inspected rendered forms are not interactive desktop acceptance.
No formal research campaign, throughput improvement, new generation law, checkpoint
migration, cloud/GPU migration or historical environment reconstruction is claimed.
Old non-1×1 scenes are rejected rather than silently resized. Historical artifacts
and running experiments are unchanged. Pre-existing CLI display/action-protocol
work in this worktree is kept separate from this scene supplement.

Hosted PR CI must verify the committed snapshot. The PR records its exact run and
final status; an earlier successful PR run does not validate this supplement.
