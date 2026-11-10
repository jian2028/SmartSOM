# Training sampling and update controls

The existing `smartsom check/run`, batch, pause/resume and evaluation commands
remain the only entry points. These settings extend the current
Scenario/Algorithm/Experiment contracts; no historical runtime is selectable.

## Experiment sampling

```yaml
sampling:
  num_envs: 1
  processes: 0
  fragment_ticks: 256
```

The defaults retain serial sampling. For two parallel environments set both
`num_envs` and `processes` to 2. Each worker owns an independent physical world,
policy sampling random stream, bound rule state and reward-component state. The
parent owns all optimizer updates. Environment count does not change role/network
parameter sharing. Native batch admission reserves workers plus the learner;
evaluation concurrency is a separate setting. Training sampling currently uses
native execution (`run.tune: off`). Custom learners keep their existing serial
contract; the new native packet-update interface is not assumed for them.

`fragment_ticks` bounds the collection interval in each environment when fragment
or transition updates are enabled. It is not a gradient batch size or an episode
termination limit. A synchronous round waits for each active environment, combines
its learning data in environment order, updates, then distributes new parameters.

## DQN learner parameters

These fields live in the existing learning policy's `learner.params`:

```yaml
update_trigger: episode       # episode or transitions
update_every: 256              # new completed transitions per network
update_amount: fixed          # fixed or ratio
updates_per_transition: 0.125  # used only by ratio mode
epochs: 3                     # minibatch updates per trigger in fixed mode
batch_size: 256                # replay samples per gradient update
target_update_interval: 100
target_update_clock: network_gradient_steps  # or global_gradient_steps
warmup_transitions: 0
exploration: constant          # constant or linear
epsilon: 0.1
epsilon_final: 0.1
epsilon_decay_transitions: 100000
```

In `episode` mode, updates are triggered when each sampled episode ends. In
`transitions` mode, completed decisions accumulate per network; thresholds are
checked at collection boundaries. Multiple crossed thresholds produce multiple
fixed update allocations. `ratio` instead accrues fractional optimizer credit
from newly inserted transitions; unused fractional credit survives checkpoints.
It is gradient updates per new transition, not replay rows per transition.

Warmup counts lifetime insertions, not current replay occupancy. Data collected
strictly below the warmup threshold do not create an optimizer backlog. Replay
capacity still bounds retained samples. Linear exploration uses each network's
insertion count, clamps at `epsilon_final`, and is unchanged by argmax evaluation.

Shared networks aggregate their owners' transitions. Independent networks each
maintain their own clocks. Agent trajectories remain separate in either case.
Automatic actions that bypass the policy do not create fictitious transitions.
An unresolved DQN action remains with its environment until the same semantic
subject makes its next decision. Fragment boundaries are not terminal states.
Episode truncation retains the existing terminal/bootstrap/omitted-tail semantics.

## PPO learner parameters

```yaml
update_trigger: episode  # episode or fragment
epochs: 3
batch_size: 256
return_estimator: gae
gamma: 1.0
gae_lambda: 0.95
```

`episode` retains complete-episode collection. `fragment` updates after each
synchronous collection round. Returns and GAE are computed separately for every
environment, episode and semantic subject, before combining the rows for one PPO
update. All samples in a round use the same policy version. Nonterminal fragments
bootstrap the critic; actual terminal episodes bootstrap zero. PPO does not use
DQN replay ratios or exploration schedules. Existing `episode_return` remains
selectable; fragment collection necessarily makes its nonterminal targets
bootstrapped finite-horizon returns.

## Compatibility and evidence

Maintained YAML files explicitly select their previous serial/episode behavior.
Absent sampling/update fields use those same defaults. Target synchronization is
now explicitly per-network by default as described below and in ADR0083. Historical run inputs and frozen source
are not modified; existing runs use their original source and identity checks.
Performance comparisons measure identical workloads, termination, decisions,
rewards and model parameters, alternating baseline/candidate at least three times.
Sampling throughput and total training wall time are different measurements;
parallel sampling is not guaranteed to speed up a CPU-saturated host.


## Durable recovery boundary

Serial training commits after a complete episode update; parallel training commits
a complete assigned episode wave. Pause, failure or wall-budget expiry discards
uncommitted work and resumes from the last committed boundary. It does not resume
a simulator midway through a fragment. Checkpoints retain optimizer/replay state,
per-network insertion/gradient clocks, fractional credit, and each sampler's random,
rule and reward state. Physical-tick and episode evaluation milestones shorten waves
so their requested boundaries are not skipped. A restored wave uses the same
configured environment count and scientific source identity.

## Target synchronization and measured replay intensity

DQN `target_update_clock: network_gradient_steps` (the default) synchronizes each
network after its own `target_update_interval` optimizer steps.
`global_gradient_steps` synchronizes configured networks after that many optimizer
steps across all networks. Both work with shared and independent ownership and
existing role learner overrides. Maintained YAML explicitly selects the per-network
clock. The previous implicit mixed clock is not a supported mode. Changing modes
is a new scientific configuration, not a live switch for a frozen run. No old
entry point, historical checkpoint migration, or alternate experiment copy is added.

`losses/<group>/networks/<network>` exposes cumulative `inserted_transitions`,
`replay_size` (current occupancy), `gradient_updates`, actual `sampled_rows`,
`target_updates` (copies after optimization, excluding initialization), and
`replay_ratio` (sampled rows / inserted transitions; zero before any insertion).
These values use the existing local loss and W&B logging path. Counter differences
between committed reports give interval rates. Shared networks aggregate owners;
independent owners remain separate. Counts survive committed checkpoint recovery.
The ratio includes warmup insertions and counts the actual smaller batch when
replay has fewer rows than `batch_size`; configured ratio times batch size is only
a nominal intensity. No additional sampling or random draws are used for logging.
