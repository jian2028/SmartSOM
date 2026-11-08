# 0081 — Diagnostic tail provenance and explicit task-window comparison

Status: opt-in development diagnostic, not a default learner change.

Consolidates the diagnostic branch record
[original ADR0077](0077-v2-diagnostic-window-objective.md), whose number overlaps
the independently integrated resource-control ADR0077. The historical record
is retained; this record is the current diagnostic reference.

Extends ADR0074 without replacing its default unresolved-tail omission. The runtime
sets RewardTimeline.end_reason from its final classified report, after distinguishing
all_delivered, drain_deadline, training_budget_cut and wall_clock_interrupted.
No physical status, reward, candidate or native DQN transition changes.

The V2 diagnostic package compares the original learner with a separate custom
learner that treats only drain_deadline as the end of an explicitly finite task
window. This is a different learning objective, not proof that safety truncation
was a bug. Natural completion remains terminal; external/budget cuts retain omission.
Blocked resource flags alone never imply absorbing termination: future arrivals,
repairs or legal transports may still permit progress.

Both arms keep the original partially observed stationary policy, with no new
remaining-window features. Thus this is not a fully observed finite-horizon Q_t
claim. Shared ownership, reward, native replay backend and optimizer settings stay
matched. Tail audit reports per-subject returns, which overlap and must not be
summed as global lost reward. Episodes 0 and every 50 are evaluated on one fixed
12-case panel distinct from the final 80-case test panel. Initial models are not
trained before their baseline evaluation. No test data selects hyperparameters.

The user authorized source changes on the current branch when custom components
need a runtime contract, and short/1000-episode diagnostic runs. Existing frozen
runs remain on their original sources. Quality objectives and social/budget
comparisons are gated subsequent studies, not silently pooled with this contrast.
