# 0085 — V3 panel, model selection and first launch

Status: user-confirmed, 2026-11-11. Supersedes ADR0084's panel counts and best ranking.

Each learned recipe authors training seeds 101–105; the first launch selects only
101 through the existing execution-only seed selector. Small on Mac selects
Base/SL shared. Medium on BudgetVM selects Base/SL shared and independent.
Both hosts use W&B online. Central retains the previous PPO algorithm with the
new workload/reward/deadline contract but is authored only, as is Large.

Validation and test each have 50 seeds per breakdown condition, 200 worlds total
per panel. Seed-count shorthand uses the established separate validation and test
domains. Validation remains every 50 episodes. Test does not select checkpoints.

The opt-in argmax_delivery_cost selector ranks fewer unfinished worlds, higher
mean delivered fraction, lower mean observed cumulative learning cost including
task-deadline penalties, then earlier episode. Completed makespan/tardiness remain
undefined for truncated cases; no incomplete case is dropped from selection.
Legacy selectors retain their historical behavior.

Each map also defines Rule slow/normal/fast and Random on the same complete test
panel. Rule's processing_mode component parameter defaults to normal, preserving
its old algorithm. Only its machine-speed preference differs among the three.
All methods on a map use the same new Rule-calibrated drain limit.

best_last retains a rolling complete recovery generation and validation-selected
best. Formal source identity and runtime admission remain separate requirements;
the selected inventory does not imply all tasks execute simultaneously.
