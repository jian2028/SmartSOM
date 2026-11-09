# 0082 — Native sampling and per-network update controls

Status: implementation under review; not a formal experimental result.

Extends ADR0074's complete-episode learning and ADR0077's one-CPU training lease
for opt-in native sampling. The public entry points, physical simulator,
observations, semantic identities, reward contract and ownership remain unchanged.
Historical composable/framework runtimes are not restored.

The existing episode runtime exposes a committed-physical-boundary callback.
Native DQN can insert completed decision transitions and update before episode
end. Native PPO can collect bootstrapped fragments from the same policy version.
Parallel workers own physical worlds and local state; the parent alone updates
models. PPO returns are prepared per trajectory before pooled optimization.
Unresolved DQN tails remain in their original environment.

DQN update triggers, update amount, replay batch size, warmup and exploration
schedule are separate settings. Insertion counters and fractional gradient credit
belong to network ownership, not physical ticks. Fixed episode updates and constant
exploration remain the default. Existing maintained YAML files declare these
choices explicitly. Scientific configuration and source identities remain frozen.

Sampling CPU/memory reservations include worker processes. Evaluation retains
that training baseline until workers exit; it cannot release training capacity
while processes remain alive. Sampler failure propagates and joins owned children.

Validation covers default behavior equivalence, isolated trajectories, independent
clocks, fractional credit, official training/recovery and paired performance.
Hosted CI and review completion are reported separately.


The user confirmed complete episode/wave durability. Mid-fragment simulator
snapshotting is outside this change. On interruption the uncommitted wave restarts
from the prior generation; successful waves save per-environment sampling/component
state alongside learner/replay/optimizer state and clocks. Wall-budget expiry is a
normal bounded stop and rolls back partial in-memory updates before final reporting.
