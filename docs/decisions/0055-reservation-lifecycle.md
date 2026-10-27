# 0055 — Reservation execution lifecycle

Date: 2026-10-25
Status: User-authorized implementation; development verification only.

Supersedes ADR0054's foreground-only lifecycle restriction. Scientific simulator,
actor/action and reward contracts are unchanged. The four author files compile in
memory through shared validators. Execution persists canonical inputs plus referenced
calibration/checkpoint evidence before any detached process is started.

The existing background launcher owns PID/birth verification, launch locks, startup
handshake, logs and cooperative stop. A new schema dispatch is an adapter, not a
second daemon. `smartsom resume` validates source and frozen inputs first. Completed
runs are idempotent; unfinished smoke/calibration/evaluation cannot yet resume and
fail explicitly rather than replaying with a fresh budget.

`ReservationSession` owns the common native/legacy training loop. A complete rollout
and PPO update commits a generation containing model, optimizer, policy RNG,
condition RNG, episode records and physical tick counters. The pointer is published
only after payloads are durable. Interrupted uncommitted rollouts are replayed;
this is not mid-tick continuation. The original wall deadline remains authoritative,
including downtime and replay. Attempts are recorded separately; only committed
physical ticks count toward the frozen learning budget. Automatic exception recovery
is bounded; explicit stop and invalid contract evidence are never auto-retried.
A killed standalone driver requires explicit `resume`; no daemon silently relaunches it.

Optional Base Tune reuses CalibrationController, ProbeSupervisor, ResourceMonitor,
ResourceBroker and the pinned Ray contract. Only CPU numerical threads and independent
seed concurrency are calibrated. One environment, rollout/update semantics, frozen
seeds, rewards and Mover stay fixed. Ray trials use the same native session and its
local durable generations; trial restoration must not roll them back to an older
reporting checkpoint. These local checkpoints are not portable to another filesystem.
`recommend` measures only; `auto` runs independent seeds with a measured fixed profile.
This adapter does not claim the composable contract's live elastic resource-changing
scheduler, GPU execution, or a new hyperparameter-search algorithm.
