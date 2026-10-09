# V3 dispatch and DQN cadence repair

Development repair based on integrated main `26245ee`. This is engineering
verification, not a new experiment, milestone acceptance or policy performance claim.

See [ADR0037](decisions/0037-v3-service-and-learning-liveness.md)
for the superseding dispatch decision.

## Dispatch boundaries

An arrived loaded vehicle with a legal destination, an available slot bound to
its actual port and queue admission retains that destination for this boundary.
The existing one-tick drop service runs before another optional Dispatcher choice.
Actual Machine START choices resolve before final Dispatcher requests, without
advancing time. The Dispatcher observes actual released capacity. Source supply
was frozen before START and retains its next-boundary eligibility rule. Semantic
replay uses the same phases. Competing ports share capacity counters, arrival
order and identity-seeded ties; a zero-time new arrival
cannot displace a previously admitted drop. Full, incompatible or queue-blocked
vehicles retain legal redirect choices. No destination inventory is reserved
while travelling.

An unloaded vehicle committed to a source keeps that commitment while its owner
has work. While travelling, the original one-shot empty-episode decision remains;
a same-target selection consumes its latch until work reappears. At each boundary where it is
arrived and the owner remains empty (PRE, POST and internal processing all empty),
it can reconsider. This is a persistent condition, not a periodic timeout.
Choosing the exact same target preserves position, arrival order and route state.
Choosing a different port of the same owner remains a real reroute. More decisions
do not guarantee that a greedy policy chooses a useful target.

The simulator remains V3. Physical metadata changes dispatch semantics from
`nonexclusive-intentions/1` to `nonexclusive-intentions/3` (also rejecting the
unpublished initial repair `/2`); old weights and
continuations are incompatible and require retraining. Historical frozen source
and recordings retain their original behavior.

## Training audit and repair

DQN uses a global learner parameter set, with separate replay stores per group.
It requires `len(replay) >= batch_size`, and samples without replacement.
Configuration now rejects `replay_capacity < batch_size`, which could never
meet this threshold. Warmup,
optimization interval, epsilon and target clocks use aggregate physical ticks.
Machine groups with 23–54 samples cannot optimize with batch 64. Reducing batch
cannot help a group with zero samples. These facts do not establish a learner bug.
There is no existing per-group batch override. A future coordinated Mac/Windows
Batch04 recipe should choose batch and collection budgets together and check each
group's decisions, replay samples and optimizer steps. Active runs are unchanged.

A separate generic bug occurs with parallel sampling: a full wave can skip an
optimizer tick (three environments and interval 16 can jump from 15 to 18).
DQN waves now stop at the next optimization or per-group target-copy boundary, preserving stable environment
order and a fixed weight version per wave. Batch size, sampling distribution,
replay thresholds, reward, bootstrap and termination semantics are unchanged.
Single-environment Batch04 was not affected by this cadence bug. Serial and
parallel optimization cadence now agrees at the configured ticks; different
wave data availability still means their learned trajectories need not agree.

At time-limit truncation, pending DQN intervals are emitted only when the staged
next boundary supplies an actual next decision input. Otherwise they are counted
as `censored_truncations`; no artificial WAIT or terminal target is invented.
This repair does not change that contract.

## Local engineering verification

At initial commit `978fe3a`, before independent-review corrections, the isolated
locked base environment passed `uv run --no-sync pytest -q`:
2,064 passed, 327 optional-dependency skips. Repository Ruff lint and format checks
passed. The existing optional learning environment separately passed 75 focused
checks covering composable learning, collection/replay, physical service, matrix
transport, persistent intent and in-memory continuation. Numerical threads were
limited to one. No formal experiments or Windows validation were performed.
