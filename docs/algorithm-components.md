# Algorithm components and action ablations

The current four inputs remain Factory, Workload, Algorithm and Experiment.
Algorithm uses `schema: smartsom.algorithm-components/v2`. A factory is an ordinary Python
function: `target` is its module-qualified name; `path` plus `factory` refers to
an external file relative to the Algorithm YAML. `check` shows the resolved source.

## Base PPO, SL and DQN

```yaml
schema: smartsom.algorithm-components/v2
action_representation:
  default: direct # required: direct or rules
control: base
production:
  default: main
  policies:
    main:
      kind: learning
      learner:
        target: smartsom.learning.ppo.create_learner
        params:
          learning_rate: 0.0003
          epochs: 3
      observation:
        target: smartsom.representations.base.observation.create_observation
      encoder:
        target: smartsom.representations.base.encoder.create_encoder
traffic:
  mover:
    target: smartsom.traffic.rule_mover.create_mover
decision_protocol:
  pre:
    accept:
      allow_stop: true
  post:
    accept:
      allow_stop: true
  dispatcher:
    pickup:
      allow_empty_wait: false
```

PPO uses episode return-to-go with gamma=1 and a separate critic channel. DQN is
masked **Double DQN**, with replay and target-network state. Change the learner
target to `smartsom.learning.dqn.create_learner` to select it. Base observation and
encoder implementations are reusable; each experiment creates its own weights.

For SL, select `smartsom.representations.social.observation.create_observation`
with `params: {arm: all}` and
`smartsom.representations.social.encoder.create_encoder` with
`params: {pool: attention}`. Existing role/stage arms and `mean-max` pooling remain
available. `arm: base` disables social inputs without changing Base operational
information. Complete recipes are `ppo.yaml`, `dqn.yaml`, `sl-ppo.yaml` and
`sl-dqn.yaml` under `configs/base-diagnostic/`. Experiment references one recipe;
Factory and Workload can be reused without copying.

Changing `pre.accept.allow_stop` or `post.accept.allow_stop` creates an action-space
ablation, not a new learner file. STOP ends the current acceptance loop. It does
not delay service by a tick. No legal candidate means natural waiting regardless
of this setting. `allow_empty_wait` concerns unloaded Dispatcher pickup, not Mover.
Rule excludes acceptance STOP even when offered; Random samples from the offered
candidates. Compare algorithms under matched action permissions.

## Mixing production strategies and fixed traffic

Add another policy and bind only the desired decisions:

```yaml
production:
  default: main
  bindings:
    dispatcher.pickup: dispatch_rule
  policies:
    main: # complete learning declaration from above
      kind: learning
      learner: {target: smartsom.learning.ppo.create_learner}
      observation: {target: smartsom.representations.base.observation.create_observation}
      encoder: {target: smartsom.representations.base.encoder.create_encoder}
    dispatch_rule:
      kind: policy
      implementation: {target: smartsom.policies.rule.create_policy}
```

Supported distributed bindings are `machine.process`, `dispatcher.pickup`,
`dispatcher.delivery`, `pre.accept`, `pre.service`, `post.accept`, `post.service`,
`input.accept`, `input.service`. All instances of a role use its binding. One
learning instance may serve several roles; their existing parameter-sharing rules
are retained. Random uses `smartsom.policies.random.create_policy`.

The shared `traffic.mover` stays unchanged in Random, Rule, Base and SL comparisons.
It jointly chooses movement/wait/yield actions without changing production targets.
Centralized mode uses `control: centralized`, `central.assign`, and rule/random or
custom policies. Omit the distributed `pre`, `post`, and `dispatcher` action blocks
there; shared scaling and quality penalty parameters remain available under
`decision_protocol`. Centralized DRL is not supplied.

For smoke/evaluation, `run.methods: [configured]` uses authored bindings. Named
`rule`/`random` comparison recipes replace production only and preserve traffic and
action permissions. A named learner evaluation must match the configured learner;
use `configured` for custom learners and mixed learning policies.

## Extension interfaces

All factories are startup-bound trusted Python code, not hot-loaded plugins.
File examples are in `examples/current/`:

- Policy: factory `(seed, control, **params)`; `choose(request, critic=None)` returns
  a semantic candidate ID (built-ins may return the corresponding semantic tuple).
  `get_state()`/`set_state(state)` preserve JSON-compatible state.
- Observation: factory returns `(public_facts, decision) -> inputs`; declare
  `factory.OUTPUT_SCHEMA`. The runtime freezes inputs and retains the original
  candidates. Built-in identity projections return `None` to reuse prepared fields.
- Encoder: factory `(group, options, **params)` creates a Torch module returning
  candidate and context features. Declare `INPUT_SCHEMA`, `OUTPUT_SCHEMA`
  (`smartsom.candidate-features/v1` for built-in heads), and a `pack` function.
  The module exposes its feature `width`.
- Full network: use `network` instead of `encoder`. Factory
  `(group, options, value_head, **params)` returns the existing actor/critic network
  contract. Declare `INPUT_SCHEMA` and `pack(requests, critics, options)`. Actor
  output masks padding; value access retains the separate critic channel.
- Learner: factory `(options, seed, training, selection, numerical_threads,
  **params)` declares `CONTRACT = "smartsom.learner/v1"`. Implement `choose`,
  `needs_critic`, `update(rewards, terminal, final_critic)`, `get_state`/`set_state`,
  `sampling_state`/`load_sampling_state`, `seed_sampling`, `discard_rollout`, and
  expose `updates`, `last_losses`, `records`. Records refer to semantic decisions
  and physical ticks. The runtime supplies the full physical reward timeline,
  including intervals controlled by rules. State includes all optimizer, target,
  replay, random and counter information needed by that learner.
- Mover: factory `(routes, seed, **params)` declares `smartsom.mover/v1`; implement
  `moves(traffic_view, tick)`, `get_state`, `set_state`. Output is a joint mapping of
  vehicle identities to legal movement actions. State restarts with the episode;
  complete-update recovery replays interrupted episodes from the saved boundary.

Policy factories declare `CONTRACT = "smartsom.policy/v1"`. Constructor context
cannot be overwritten through `params`. Interfaces are keyword-based; no adapter
for arbitrary third-party training loops is implied. Network forwards retain eval
mode during rollout and updates; gradients remain enabled for optimization.

For a multi-file extension:

```yaml
implementation:
  path: ../../research/my_policy/entry.py
  factory: create_policy
  source_root: ../../research/my_policy
```

Use `from .helper import ...` for local imports. The compiler freezes the declared
root, preserving package structure and resources; do not point it at an entire
home or run directory. Installed third-party dependencies are recorded separately
in the environment. Changing a frozen helper file must fail identity validation.

## Migration and recovery

```sh
smartsom config migrate old/algorithm.yaml --output migrated-algorithm
smartsom config migrate old/experiment.yaml --output migrated-experiment
smartsom check migrated-experiment/experiment.yaml
```

Migration writes a new directory and `migration.json`; existing destinations are
rejected. Legacy source pins are checked before importing Python; output paths
retain their original location. Matrix variants carry their own required files,
and legacy STOP axes remain coupled. Unknown historical schemas or incompatible extension interfaces require
manual adaptation using their archived evidence. Migration never rewrites Python
algorithms or silently translates model weights. Old runs continue with their
matching frozen source, lockfile and original extension implementation.

Scientific edits require a new run. Operational monitoring/concurrency changes
retain the existing resume rules. Full current checkpoints include component and
scientific identity; matching tensor dimensions alone do not establish compatibility.

## Learning diagnostics

Existing `actor`, `critic`, and `q` loss fields still describe the final minibatch.
`update_diagnostics` additionally reports sample-visit weighted means and counts
for each role and role.stage. PPO actor loss, entropy, approximate KL and clip
fraction exclude forced single-candidate decisions; critic loss includes every
training sample. DQN Q loss, absolute TD error and selected-action Q values count
every replay visit, including repeated visits to an older transition. Epsilon and
target-network update counts are states at the end of the complete learner update.

These diagnostics reuse existing forward outputs and do not sample actions or
change losses, optimizer steps or checkpoint state. A missing group or metric
means no eligible samples, not a measured zero. Stage means must not be averaged
without their corresponding sample counts.
Non-finite diagnostic values produce a missing mean and an explicit non-finite
visit count, rather than JSON Infinity or an average silently excluding those visits.


## Direct actions and learned rule actions

Every Algorithm explicitly declares `action_representation.default: direct|rules`.
Role and role.stage overrides are independent of policy binding, Base/SL, learner,
network sharing and Mover. `check --verbose` and `--json` display expanded modes.

```yaml
action_representation:
  default: direct
  roles:
    dispatcher: rules
  stages:
    dispatcher.delivery: direct
  rules:
    dispatcher.pickup: [nearest, most_ready_stock, fewest_commitments]
```

An omitted subset uses the built-in catalog. List order is alias priority; builtin
rule codes retain their catalog identity. STOP/WAIT permissions stay in
`decision_protocol` and are never merged with ordinary rules. A subset with no
available representative and no enabled special action raises an actionable error.

Custom rule declarations use `custom_rules.NAME: {path, factory, source_root}`,
with `NAME` explicitly included in a role/stage subset. The factory declares
`CONTRACT = "smartsom.rule/v1"` and returns a deterministic stateless callable
`(facts, decision, candidates) -> candidate_identity | None`. The request supplies
common fields, without social-only sidecars. Public facts and candidates are
immutable; future worlds/hidden quality are excluded. File and multi-file source
freezing follow the same rules as other components. A custom rule cannot return an
arbitrary core action. Use a policy component for stateful or stochastic logic.

This is discrete named-rule selection, not continuous rule weights. Rule aliases
can reduce action coverage; availability, alias and selection rates have explicit
opportunity/decision denominators. Full per-decision audit maps remain local when
recording is enabled and are excluded from learning replay memory.

## Replay recording

```yaml
# Experiment YAML; all three default to off
replay:
  train: off
  validation: first
  evaluation: all
```

`first` records the first episode per condition within a training trial, or the
first case per method and condition within each checkpoint/decoder evaluation
panel. `all` records each. Paths are `recordings/train/episode-N/` in a trial and
`recordings/case-N/` in an evaluation panel, with a separate `attempt-N/` for each interrupted retry, containing `decisions.jsonl` and a
Studio-compatible `replay/`. Counts/summary diagnostics do not require full replay.
Training defaults to summary metrics; evaluation's detailed metrics are separate
from full replay and from W&B upload policy.

Current POST credit excludes unstarted PRE work. V1 YAML must be migrated explicitly
with `smartsom config migrate SOURCE --output DEST`; migration sets the historical
implicit representation to `direct` and reports changed current protocol semantics.
It creates a new experiment, not a compatible continuation of an old checkpoint.

## Prespecified checkpoint evaluation and operating limits

Final test panels may name saved episode checkpoints independently of validation.
They execute after training completes and never select the best model:

```yaml
train_episodes: 1000
final_evaluation: true
evaluation:
  checkpoints:
    - episode: 0
      modes: [sample]
    - episode: 1000
      modes: [sample, argmax]
  case_seconds: {sample: 600, argmax: 1200} # optional wall protection per case
run:
  task: train
  seconds: null # explicit: no wall-clock stop; physical/episode budgets still apply
```

Each listed checkpoint is retained at `checkpoints/episode-N.pt`. Its test panels
live under `evaluation/episode-N/DECODER`, with the checkpoint's actual training
progress. Unlisted default behavior still evaluates selected best/last once.
`sample` means policy sampling for PPO and epsilon-greedy for DQN; `argmax` means
policy argmax or greedy Q respectively. Results also record the explicit `decoder`
label; these are not equivalent stochastic policies.

`seconds: null` is supported for native train/evaluate. Smoke, calibration, Tune
and recovery-calibration protocols retain required finite wall budgets. Case wall
interruptions remain censored evidence, not successful drainage. A per-case limit
never shortens the scientific physical horizon or prevents later cases being tried.

### Archived rule-DQN initialization

For a new run matching the archived rule-DQN initialization stream, the builtin
DQN learner accepts `initialization: legacy-rule-dqn-v1` in its `params`.
The default is `native`. Compatibility mode requires builtin shared-role
Base/ALL attention models without role-specific parameters or a custom network.
It constructs the historical frozen critic and masked wait head in all five
roles before social heads; the critic remains outside TD loss and Adam. Although
WAIT is disabled, retaining its zero-gradient forward preserves Adam state and
Torch RNG consumption. For the historical configuration in which all five
resolved role reward scales are exactly `0.001`, this legacy preset also preserves
the archived arithmetic order: divide each reward by 1000, then accumulate
transition suffixes. This avoids binary64 replay rounding differences from
multiplying by `0.001`; the input reward sequence is not modified. Native mode
and other scale configurations retain their existing multiplication path.
The option does not enable WAIT, change the mathematical reward or rules, or
import old checkpoints. It is part of scientific/checkpoint identity. World
streams, reward scale, decoder, budget and test panels still need explicit
configuration; this compatibility preset alone does not prove full equivalence.

### Checkpoint and launch implementation details

Checkpoint writes use pickle protocol 5; the loader still accepts earlier
protocol 2 checkpoints. Builtin DQN observation replay is serialized as one
lossless Python object graph inside the Torch checkpoint and loads as a dictionary.
That serialization layer retains its supplied values and references. Separately,
builtin DQN records and replay use lossless compressed row storage, described in
[ADR0068](decisions/0068-dqn-resident-packed-storage.md). It retains every field,
numeric precision and within-row graph, but does not preserve object identity
across separate row reads. Sampling order and learning arithmetic stay unchanged.
New packed checkpoints require the new reader; compatible older raw checkpoints
can be read without mutating their supplied payload. Custom observation components
retain ordinary Torch serialization, including tensor device remapping, and
unsupported mutable/custom rows retain raw storage. These checkpoints remain
trusted local Python pickle artifacts.

Within one launch, validated typed inputs can be reused only after rechecking
their frozen content hashes and execution overrides. Component resolution reuses
one source snapshot within a construction operation and checks the live bytes
again on exit. Installed-package metadata is shared only within that launch;
source bytes and Git identity are still checked at the existing boundaries.
The two complete builtin source archives are independent copies, never links to
the mutable checkout. Both archives are listed in the plan with content hashes
and rechecked on load. Each identity boundary obtains HEAD, branch and complete status in one
live Git porcelain-v2 snapshot; its raw status and format marker are retained
as audit evidence. This changes the status text format, not the frozen commit
comparison or source-content verification. Process control caches the macOS library handle, while querying
each PID's current birth identity separately.

These changes do not establish a performance or full-replay memory guarantee.
The current requested performance check uses 1280 ticks, with Base/ALL each
compared against the archived driver in five paired measurements. Report total
and one-time startup separately. The 5% limit applies after startup, including
episode generation, rollout/update, logging, metrics, checkpoint durability and
final save/close. Repeated runtime work cannot be excluded as startup. Earlier
128-tick measurements are historical diagnostics, not this acceptance result.
Production resource acceptance requires a separate full-capacity replay
measurement.
