# 0044 — Training passing potential with terminal closure

Date: 2026-10-12
Status: User-authorized training conversion; qualification pending.

This extends ADR0039 only for an explicitly selected passing-rate training
conversion. The original shipment task objective, evaluation, best selection,
throughput/tardiness terms, observations and default rewards remain unchanged.
The new `smartsom.passing-rate-training/v1` selector declares a positive
`pseudocount` and fixed `objective_episode` terminal compensation. It requires
the shipment task and gamma1; it cannot stack a team/role reward extension.

For pseudocount9, private training uses Psmooth=G/(S+9) and emits the configured
passing weight times its committed interval difference. At the genuine objective
episode end it adds that weight times (Ptrue-Psmooth). Ptrue is computationally
zero when S=0, while its display may remain undefined. Simultaneous shipments
use whole-batch counts. From an empty episode, the sum of quality feedback is
the original weight times final G/S. Learning trajectories can change; model
equality between old and new training is not a requirement.

The conversion executes in TrainingSession._finish_tick before reward scaling
and PhysicalCollector.collect. A unique physical environment/episode ledger
records smoothed feedback, compensation and the original objective separately;
group copies are never summed to claim closure. A task's window end is a true
objective terminal even if simulator status is truncated. Existing PPO zero
terminal bootstrap and pending DQN terminal flush apply, including owners with
no decision on the last tick. Ordinary non-task time-limit bootstrap is retained.

An update/drain, validation, checkpoint or cooperative pause is not an episode
end. A total training budget ending inside an episode leaves the potential gap
pending and pays no fabricated compensation. The ledger survives FULL restore,
including pre-/post-terminal boundaries, and rejects changed contracts or totals
before mutable learner state loads. Selected configuration, scientific identity,
model/run provenance and continuation include the version and terminal semantics.
Oracle counts remain private reward inputs and are never policy observations.

Formal execution requires integrated main, CI, independent review and separately
qualified reward closure/engineering evidence. Existing experiments are preserved.
