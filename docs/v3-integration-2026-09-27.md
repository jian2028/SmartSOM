# Composable v3 dependency integration

This integrates the existing composable-policy work into the cleaned repository.
The original active worktree, source identity, environment and study remain
unchanged. No new first Small configuration or formal training run is created.

The dependency closure includes staged physics, semantic replay, travel-time
providers, bounded workload content, classified v3 recipes, PPO/DQN continuation
and study preparation/execution. Studio inspector additions remain outside this
integration. Machine processing-rate multipliers are included; an unused proposed
factory-level multiplier is not included.

Existing engineering examples move from `configs/<category>/` to
`configs/test/<category>/`, including policies, compositions, rules and transport.
Relative references preserve the category relationships. Example outputs resolve
to repository `runs/`. Daily `configs/factories/large.yaml` and bundled Studio
templates are preserved. Historical verification commands in
`composable-verification.md` and `small-hv-verification.md` remain frozen; use the
path mapping above to locate their current authoring counterparts.

Fresh base quality check: 1452 passed, 235 optional-dependency skips.
Focused real learning/continuation and parallel-study checks: 47 passed initially,
then 19 passed including the shared-memory retry. Configuration/API/example
regressions: 110 passed. Ruff and formatting passed.

The broader legacy learning acceptance was interrupted after 154 passes and one
failure: its fixed paired-run report lacks execution_replay for some failed
evaluations (run did not complete). This is not recorded as a passing acceptance
check. A local engineering test does not establish research-performance or
CUDA/CARC claims.
