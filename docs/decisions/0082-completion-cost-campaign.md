# 0082 — Opt-in complete-order episodes and reward-based selection

Date: 2026-11-09. Status: user-authorized V3 implementation.

Extends ADR0079/0080's periodic selection and ADR0081's end-reason contracts.
Existing V2 defaults and finite-window diagnostics remain unchanged.

`episode_end: all_delivered` requests finite-order completion in training and
both periodic/final evaluation. Runtime extends the seeded scenario before the
next physical step would reach its drain limit. Extension preserves all existing
physical inputs and the prefix of outages; future outages continue with the same
seed. No intermediate boundary is a learning terminal. All original demands must
ship, including scheduled future arrivals. The last committed output shipment is
the terminal time. External/budget interruptions are not successful completion.
`drain_ticks` controls extension chunk size in this mode, not a new larger cap.
Imported scenarios must match their generative inputs before they can extend.
Recorded manifests retain the extended scenario for deterministic replay.

The completion reward component uses fixed original demand count N and costs:
`-(elapsed + overdue_delta/N + quality_penalty*failed_delta/N)/scale` with explicit
makespan/tardiness weights. V3 selects weights 1/1, quality_penalty1000, scale1000,
gamma1 and learner scale1. Per-interval outcomes are incremented once; no rolling
failure-rate denominator, clipping or independent per-component normalization.
Quality remains trainer-only; actors gain no new privileged observations.

`learning_cost` is negative accumulated team reward, available for completed
cases only. `periodic_evaluation.cost_metric` defaults to historical joint_cost
and permits learning_cost for custom objectives. Eligibility and tie order remain
unchanged; selection metadata records the metric. Raw V2 joint_cost is retained,
not relabeled as V3's objective. No final test case selects a checkpoint.

V3 pickup is a four-rule subset with HPOF's resident-backlog tie-break. V2 actor
feature dimensions and semantic rule identifiers remain unchanged to control
network architecture. The V3 dataset includes all three maps; user authorizes
only Small/Mac and Medium/BudgetVM execution and excludes Morris from launches.
