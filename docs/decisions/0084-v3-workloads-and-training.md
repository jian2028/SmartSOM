# 0084 — V3 workload and native training preparation

Status: user-confirmed design; implementation verification recorded separately.
Date: 2026-11-10.

Extends the current component, evaluation and native sampling contracts, including
ADR0082/0083. Port the previous V3 completion-cost, task-deadline and validation/test
semantics onto the integrated runtime, preserving PR41's sampling and target clocks.
Do not restore an alternate runtime or resume historical checkpoints under new code.

The user confirmed 3–6 distinct operation types per job, uniformly distributed,
eight fixed templates per length, A-H types, no repetitions, existing continuous
±20% work variation followed by mode scaling and ceiling, and unchanged order
counts/initial orders/exponential arrival intervals. Small retains eight machines;
Medium retains twenty. Each machine supports two types with balanced type coverage.
Retain other production settings and existing machine capability pair assignments.

DQN uses per-network transition thresholds 128, update credit 1/128 per insertion,
batch 256, warmup 1024, network-gradient target clock interval100, replay10000,
learning rate0.0003, gamma1 and linear epsilon0.30→0.05. Role/map decay lengths
and the two-environment/256tick collection candidate are explicit in V3 inputs.
Learning budgets follow network ownership; independent agents never share replay
or optimizer state because multiple environments are collecting experience.

General completion reward and drain_deadline_terminal are opt-in. A true deadline
adds the configured unfinished-task cost and disables terminal bootstrap; fragments
must retain unresolved DQN actions and have neither terminal cost nor false done.
PPO batched rollouts must apply this rule separately to each environment's timeline.
Parallel workers receive the explicit episode-end mode. Durability remains at
complete episode/wave boundaries as defined in ADR0082, not mid-fragment snapshots.

General validation/test count shorthand uses separate fixed seed domains, freezes
world inventories and rejects overlap. Preserve historical authored names for
historical input parsing; there remains one current runtime. V3 uses 100 worlds
per condition for each panel, validation every50episodes, evaluation workers20,
and validation-only best selection by completion eligibility and learning_cost.

The campaign recipe inventory and pending calibration are documented in
experiments/v3/README.md. New route workloads require new Rule drain calibration;
old calibrated values are not transferred as validated. Scientific settings stay
in Scenario/Algorithm/Experiment inputs. All functions remain under existing
official entry points; formal launch and publication require separate authorization.
