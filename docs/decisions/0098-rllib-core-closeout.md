# 0098 — Minimal RLlib/Tune core closeout

Status: user-authorized scope; local Linux core engineering acceptance complete.

This narrows the completion scope of ADR0089 and preserves ADR0097's representative
PPO/DQN performance gate. Ray 2.60 and the approved replay checkpoint/fragment
extensions remain as specified by ADR0093–0096. No algorithm, scheduler, exploration
or model storage implementation is added for legacy compatibility.

Complete the public training loop: scientific calibration provenance, official
updates, fixed periodic panels, best/last references, final evaluation, whole-run
pause/resume/stop, finite failure policies and basic three-objective Optuna restore.
PPO best uses the official sampled panel; DQN uses its default non-exploring panel
(ADR0091). Official iteration boundaries may overshoot physical/episode targets.

Defer standalone evaluation and per-case restore, window/fault recovery analysis,
centralized/Morris learning migration, full Rule/Random comparative experiments
and new training recordings. Keep existing rule smoke, scientific calibration and
historical readers. Public check/run share a framework-free support check and
reject unsupported configuration before Ray startup or run allocation. Scientific
tasks cannot silently ignore deferred settings.

Migrate valid lifecycle assertions to the official framework; consolidate duplicate
coverage and retire mechanism-specific native exact-wave/storage tests. Preserve
snapshots and the contract mapping in `archive/rllib-transition-tests/README.md`.
No blanket skips or weaker scientific assertions may hide missing core behavior.

Existing three-pair PPO training, PPO evaluation/best-last and DQN training timing
evidence remains valid within its frozen version. Compare the final source against
those versions. Test/documentation/startup-validation-only changes do not require
long reruns. Changes to training, evaluation, checkpoint or live publishing costs
require only the affected path's three alternating comparisons, with the same
five-percent steady and end-to-end median limits. No new model matrix or formal
1000-episode campaign is authorized.

Linux CI and independent engineering/science/testing review are required. Unverified
platforms remain explicit. Existing experiments, worktree changes and historical
evidence stay untouched; this decision authorizes no publish, merge or release.
