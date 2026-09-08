# 0011 — Centralized learning projection and episode contract

Date: 2026-09-08

Status: Accepted. Shared projection/Gym and optional framework training/checkpoint
evaluation are separate implementation checkpoints with separate acceptance evidence.

## Authority and ownership

This decision implements the approved Week2 item 12 plan. It preserves ADR 0002's
configuration ownership and ADR 0003's reveal boundary. It supersedes the older
planning note suggesting a learning action directory built from the complete
hidden instance. No simulator state, event ordering or feasibility rule changes.

`learning.projection` uses standard-library types and accepts only public factory
resources and `DecisionContext`. `learning.gymnasium` owns episode reset, rewards,
limits and Gym protocol; every physical transition calls `Simulator.step`.
Framework adapters consume that common representation, never private engine state.

## Projection v1

The algorithm declares maximum jobs, operations per job, execution modes per
operation, and fixed time/count normalization scales. Privileged preparation
checks capacity against the complete input before Simulator construction. It does
not give hidden counts, bounds or semantic IDs to the policy. Overflow is an
error, never candidate truncation.

Jobs bind after reveal, ordered by `(reveal_at, job_id)` among newly visible jobs.
Operations follow explicit predecessors; modes sort semantic IDs. Bindings never
move or recycle during an episode. Empty job/operation/mode rows are all zero.
An immutable binding table associates selected slots with true semantic actions.

The fixed Discrete blocks are dispatch, AGV transport, direct transfer and one
WaitNextEvent. Physical masks equal engine candidates, including legitimate
aggressive transport and rerouting choices. SPT admission rules are not applied.
Waiting requires a public witness: processing/paused work, a down machine, an
empty/loaded AGV, or a revealed future release. Hidden event times are not queried.

The flattened observation contains visible operation lifecycle and completed net
work, jobs and locations, selected/available execution modes, published inspection,
machine availability and holding phase, capacities, reservations by vehicle/job,
vehicle tasks and known topology. Resource IDs use sorted public factory catalogs;
job references use bound slots. Base-mode grouping and quality-mode labels remain
explicit in the binding/feature contract. Hidden probabilities have a missing flag.
No quality draws, unfinished actual work, future repairs or unrevealed jobs enter.

Ticks divide by the configured time scale. Finite capacities encode
`capacity / (count_scale + capacity)` with a separate infinity flag; counts used
as sequence identifiers divide by count scale. No normalization fits an episode.
The Gym view uses float32 and rejects nonfinite conversions. It never clips facts.
The unbounded Box emits the Gym checker's advisory bounds warnings; this does not
waive finite-value validation. Framework views share numerical values and masks:
RLlib uses `observations/action_mask`; SB3 uses observations and `action_masks()`.

## Episode lifecycle

Each reset creates a fresh Simulator. Fixed inputs do not change with Gym reset
seed. A scientific episode source can supply materialized disturbances by episode
number; the Gym RNG never owns the run's scientific seed authority. A source may
not change the frozen factory or base workload.

Successful reward is negative elapsed time, including initialization advancement
charged on the first decision. With gamma 1 the return equals negative makespan.
Default limits are 1024 decisions and 10000 ticks, checked after atomic physics.
Completion beyond the tick limit is truncated, not reported as successful.

Deadlock and no representable public action (`policy_stalled`) are failed terminal
episodes. Limits truncate. Failed return is `-max(tick_limit + 1, actual_tick)`;
there is no successful makespan for failure or truncation. An initial failed state
raises `EpisodeStartFailure` rather than giving a learner an all-zero action mask.
Terminal observations/masks are zero and never request another action.

Invalid indices leave physics untouched. The generic Gym checker mode reports a
fatal invalid-action terminal because the checker samples without masks. Actual
training must enable strict actions, which raise immediately. No substitute action,
hidden auto-routing, invalid-action training sample or silent penalty recovery.

## Training inputs, checkpoints and completion gate

RLlib PPO and SB3 Contrib MaskablePPO are optional backends. Training configuration,
episode seed recipes, bounded evidence, checkpoint manifests and inference through
existing run/study are implemented in the second checkpoint. There is no training
resume CLI, multi-agent layer or change to the makespan objective.

The training run uses schema `smartsom.training-run/v1`; algorithm settings remain
in `algorithm.yaml`. `resolve_training_run` freezes the base workload once.
`smartsom.training-episode/v1` hashes the canonical JSON array of version, root seed
and episode index with SHA-256 and takes the first eight bytes as a big-endian
integer. Existing named-seed derivation materializes the enabled generated inputs.
Fixed imports are unchanged. `smartsom.learner-rng/v1` hashes version, root seed and
provider in the same way, then maps modulo 2**31. It seeds framework RNG only.
Neither recipe changes ordinary run/study seeds. Evaluation uses the existing
study recipe with its separate root; scientific input pairing excludes algorithm.

Capacity, projection and `smartsom.learning-structure/v1` identity bind a checkpoint
to the same canonical factory/base workload, module enablement, reveal trigger
and probability visibility. Different realized disturbances are compatible;
cross-route or cross-size reuse is not promised. File digests, exact dependency
versions and changed before/after weight digests are checked before evaluation.
The exported checkpoint policy returns semantic actions through `OnlinePolicy`.

Training stores inputs and a compact ledger sufficient to regenerate every episode,
including slot bindings, actions, rewards and observation/mask/trace digests. Failed
episodes also store full traces. Sampling may end inside an episode; that record is
`training_budget_stop`, not an artificial environmental terminal. PPO sampling
budgets must be whole rollouts, with no automatic overshoot. Both framework drivers
reject nonfinite learner metrics and confirm restored parameter digests.

The pinned Ray 2.58 backend confines Trainable scratch output to the attempt and
refreshes its actual trainable-parameter counters before reduction. This avoids
empty-counter NaNs without suppressing nonfinite metrics. These version-specific
adaptations are covered by real training, not implemented in Gym or the core.

Full item 12 requires actual updates/save/load from both backends under the fixed
4096/1024-step budgets, followed by all 15 paired evaluations and exact replay.
Passing interface tests alone is not completion or a research performance claim.

Primary framework contracts: [Gymnasium termination/truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/),
[Ray 2.58.0 masking example](https://raw.githubusercontent.com/ray-project/ray/ray-2.58.0/rllib/examples/rl_modules/classes/action_masking_rlm.py),
[MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
