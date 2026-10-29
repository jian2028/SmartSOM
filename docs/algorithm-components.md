# Algorithm components and action ablations

The current four inputs remain Factory, Workload, Algorithm and Experiment.
Algorithm uses `schema: smartsom.algorithm-components/v1`. A factory is an ordinary Python
function: `target` is its module-qualified name; `path` plus `factory` refers to
an external file relative to the Algorithm YAML. `check` shows the resolved source.

## Base PPO, SL and DQN

```yaml
schema: smartsom.algorithm-components/v1
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
