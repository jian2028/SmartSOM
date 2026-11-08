# 0077 — Explicit V2 diagnostic window objective

Status: opt-in development diagnostic; default V2 behavior unchanged.

Extends ADR0074's learner input provenance. RewardTimeline.end_reason records
all_delivered, drain_deadline, training_budget_cut or wall_clock_interrupted after
runtime boundary classification. Existing native learners ignore this metadata.

The separately identified create_window_learner component treats drain_deadline
as a learning terminal, preserving the final subject reward and using zero
successor value. Budget cuts and wall interruptions retain prior semantics.
This is a finite task-window objective ablation, not a claim that the previous
safety-cut treatment violated its original specification. Physical metrics,
reward coefficients and unfinished-job censoring remain unchanged.

Both arms retain the same observations, without explicit remaining-window input.
The comparison is under existing partial observation and memoryless policies,
not a fully observed finite-horizon Q_t model. Adding time would be a separate
observation variable in both arms.

The user authorized isolated worktree implementation, fixes and staged diagnostics
through existing official commands. Generated development evidence retains source
hashes and frozen inputs; it is not promoted to formal multi-seed research evidence.
No publication, merge or historical-run migration is performed by this change.
