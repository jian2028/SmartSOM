# Algorithm components and action ablations

The authored inputs are Scenario, Algorithm and Experiment. Scenario owns the
factory and job-generation definition; internal workload models are not a second authoring interface.
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
information. Complete recipes are `base-ppo.yaml`, `base-dqn.yaml`, `sl-ppo.yaml` and
`sl-dqn.yaml` under `configs/algorithms/`. Experiment references one algorithm
and a reusable Scenario.

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
      observation: {target: smartsom.observations.rules.create_observation}
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
`decision_protocol`. Centralized PPO/Double DQN use explicit learning components; see ADR0074.
Centralized mode may set `central: {wip_limit: N}`, a CONWIP release rule: while at least
N jobs are released (on vehicles, in machines or station buffers, or assigned for
pickup at system input; arrival queue and output excluded), transports from system
input are withheld. Without it, central release can fill every station and lock
the cross-station buffer cycle. Omitting `central` means unlimited (previous behavior). The
limit is part of the centralized method specification; report it with results.
The E1a network parameter `advance_only_when_idle: true` masks ADVANCE (advance
time) while any transport or process task is legal, so the controller is
work-conserving and an argmax over the flat task list cannot idle vehicles and
machines by preferring the single ADVANCE. It removes deliberate waiting; report
it with results.

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
  candidates. Every projection is callable; no prepared-field fallback exists.
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

## Current formats and recovery

Author Scenario, Algorithm and Experiment. Standalone Factory/Workload experiment
inputs and migration commands are retired. Compilation freezes a current resolved
snapshot and its authored Scenario. Existing historical runs and checkpoints use
the source/environment in the external archive; current code has no legacy reader.

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

Current POST credit excludes unstarted PRE work. Historical V1 YAML uses archived
source. A current recipe requires an explicit action representation and creates
a new experiment, not a compatible continuation of an old checkpoint.

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

### Native initialization

Only `initialization: native` is accepted. The current Base/SL definitions retain
their observation/action/learning parameters, but native initialization is a new
experimental condition relative to historical rule-DQN runs. Do not relabel old
weights/results as this new condition.

### Checkpoint and launch implementation details

Checkpoint writes use pickle protocol 5; current checkpoints use the v4 component contract; old contracts are rejected. Builtin DQN observation replay is serialized as one
lossless Python object graph inside the Torch checkpoint and loads as a dictionary.
That serialization layer retains its supplied values and references. Separately,
builtin DQN records and replay use lossless compressed row storage, described in
[ADR0068](decisions/0068-dqn-resident-packed-storage.md). It retains every field,
numeric precision and within-row graph, but does not preserve object identity
across separate row reads. Sampling order and learning arithmetic stay unchanged.
Checkpoint source and component identity must match the current reader. Custom observation components
retain ordinary Torch serialization, including tensor device remapping, and
custom learner rows retain ordinary list storage. These checkpoints remain
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

## Morris and custom Rule controllers

[`configs/algorithms/morris.yaml`](../configs/algorithms/morris.yaml) selects one
central actor/critic through packaged observation/network/learner components.
`control: base` retains the nine reservation decision points. It is an adapted
Morris comparator, not the separate `control: centralized` direct-assignment
mechanism and not an exact reproduction of the original paper.

[`configs/algorithms/justin.yaml`](../configs/algorithms/justin.yaml) selects the
Justin heuristic component. Default Rule, Justin and custom rule functions read
named public facts. Random samples legal candidates with its restorable RNG.
There is one current research experiment family; to change methods, create a new
Experiment definition using the same Scenario/test panel and select its Algorithm.
Historical comparator experiment directories are not maintained.

Custom learners may expose `validate_parameters(**params)` and
`network_ownership(options)`. `begin_episode()` resets episode-local state;
`episode_metrics()` contributes bounded instrumentation. Native complete-update
transactions remain the durable recovery boundary.

## The observation interface

Every policy explicitly declares an observation factory. The factory must return
a callable `(facts, decision)`; returning `None` and `request_fields` flags are
not supported. `decision` contains tick, role, owner, stage, rejected-owner IDs,
and semantic candidates. Candidates contain identity, action and optional rule
code; there are no numerical context/set/feature fields on a physical decision.

```python
def create_observation(*, time_scale=200.0):
    def observe(facts, decision):
        return {
            "clock": (facts["tick"] / time_scale,),
            "candidate_count": (len(decision.candidates),),
        }

    return observe


create_observation.OUTPUT_SCHEMA = "example.clock/v1"
```

Named dict/list outputs are deeply frozen, including directly constructed
`Facts`; mapping keys must be strings. The network reads `request.inputs`.
Trainable encoders belong in the network, not in the observation component.

Public queries are valid during the synchronous observation/rule callback.
Queried rows are immutable snapshots. Do not retain a query provider for later
use: materialize the fields needed in the returned inputs. A new callback always
sees changes made by previous decisions, including changes within the same tick.

Local queries include `facts.jobs[id]`, `facts.agvs[id]`,
`facts.machine_state[id]`, `inventory(owner)`, `ready(owner)`, `quota(owner)`,
`commitments(owner)`, `nominal(job, machine, mode)`, and current public work.
These expose only released jobs and current public quantities. `facts["jobs"]`,
`["machines"]`, `["vehicles"]`, `["buffers"]`, `["completed"]` and `["topology"]`
materialize a facility snapshot for components such as Morris. Base/SL use local
queries and do not construct that full snapshot. Information scope is the
observation author's explicit choice, not an assertion that each agent sees all
facility state.

Base returns immutable `BaseInputs(context, sets, candidates, social)`. These are
component-owned model inputs; the encoder's 48-column layout remains unchanged.
SL uses the same Base builder and appends candidate-associated social links.
Rule resolution runs on named scores before observation construction, retaining
legal actions, alias priority, semantic tie-breaking, STOP and EMPTY_WAIT.
The SL component uses the full physical candidate set when constructing STOP
relations, then maps the selected representatives by physical action.

The editable [`examples/current/compact/`](../examples/current/compact/) package
illustrates Python feature definitions, observation statistics and a matching
network. Its 4/6 scalar inputs are a teaching example, not the current research
Base/SL replacement. Change only YAML for component selection/parameters; change
the component Python code for new fields, statistics or network logic.

Global observers may use `facts.snapshot()` to obtain the immutable full public
view once per decision. Local observers should query only the rows they need.
This shares an already frozen value; it does not cache dynamic state by tick.


## Custom training components (ADR0074)

`production.policies.<name>.critic_observation` uses the ordinary component factory
and returns `(facts, LearningContext) -> immutable inputs`. Context includes tick,
group, owner, stage and bootstrap, never synthetic action candidates. Full networks
may declare `CRITIC_SCHEMA` and implement `value(inputs)` to pack named value inputs;
rollout `forward` and bootstrap must use the same value computation. Builtin vector
critics use the existing vector contract. DQN does not evaluate a critic projection.

`reward` selects a factory `(orders, quality_penalty, **params)` declaring
`smartsom.reward/v1`. Its object implements `__call__(RewardTransition)`,
`begin_episode()`, `get_state()` and `set_state(state)`. State is saved at complete
update boundaries. begin_episode resets episode-local state; durable accumulators
are the component author's explicit choice. The callback returns
`{"team": number, "roles": {role: number}, "agents": {(role, owner): number}}`.
Missing terms are zero; each subject gets team + its role + its agent term.
All values must be finite. Reward state and outputs never enter Actor/Critic inputs.

The transition holds start/end physical ticks, committed semantic actions, public
before/after views, events, outcomes and terminated/truncated flags. Before is
sampled after zero-time choices, before physical advance. Outcomes contain only
committed shipments, passing-rate change, overdue integral and failed shipments.
A reward may set `public_snapshots=False`, `requires_events=False`, and
`requires_actions=False` to omit data it does not use. Unspecified flags default
to requesting the data. Default and E1a rewards do not consume action records. It must account for end-start and must not depend on recording.
Idle physical intervals may span multiple ticks without an intermediate event.

`learning_return`, `role_returns` and `agent_returns` are separate from physical
metrics and `observed_return` (the standard diagnostic objective). Role/agent
report values are additive components, not duplicated team totals. Recordings store
reward intervals in rewards.jsonl plus scientific component identity; playback never
re-executes a reward plugin.

PPO params select `return_estimator: episode_return | gae`, `gamma` and `gae_lambda`.
Default episode_return/gamma=1 preserves prior configurations. Discount is per
physical tick; GAE lambda is per consecutive decision by the same subject. Native
Double DQN uses the same physical-time discount and per-subject rewards. Custom
learners receive RewardTimeline and may call
`learning.transitions.decision_transitions(records, rewards, terminal=..., gamma=...)`.
Central DQN requests a read-only final candidate snapshot via `needs_next_request=True`;
PPO bootstrap does not evaluate an Actor. Shared parameters never merge agent paths.

`configs/algorithms/e1a-current-ppo.yaml` is the E1a-current adaptation; the dqn
sibling is a separate engineering comparison. `examples/current/e1a-current-*.yaml`
are bounded smoke recipes, not research results. Both use the current direct tasks
and mover. Actor features combine the unchanged Morris global public vector with
15 candidate scalars and three owner-association one-hot vectors (machine, vehicle,
destination). Critic sees the same global vector; both networks have two 64-unit
tanh layers. Fixed-map identity is checked, including maps with equal dimensions.

E1a-current reward is .5*shipments/128 + .1*passing_change
- 2*overdue_time/(128*4096) - starved_machine_ticks/(machine_count*4096).
All constants are explicit component parameters. Starvation counts available idle
machines with empty input and compatible next-operation jobs waiting elsewhere,
excluding system input/output, processing jobs and publicly failed jobs. It is
measured after assignments over the physical interval. E1a-current uses full-episode
GAE updates; remote E1a used fixed fragments and a different dispatch/mover mechanism.

Committed reward actions carry `tick`, `group`, `owner`, `stage`, semantic
`identity`, and the executed `action`. Reward interval totals are placed at the
interval's final physical tick in the learner timeline (interior entries are zero),
preserving the existing default time-discount convention. Recorded
`Playback.learning_return(tick, group=..., owner=...)` reads these committed totals
without executing plugin code; physical `observed_return` remains a separate metric.
For an interval `[start, end)` with total `R`, its contribution from a decision
at `start` is `gamma**(end-start-1) * R`, with boundary value discounted by
`gamma**(end-start)`. Playback includes it only at `tick >= end`; it does not
interpolate partial rewards inside the interval.
