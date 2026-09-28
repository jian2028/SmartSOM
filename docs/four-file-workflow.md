# Four-file experiment workflow

Daily authoring uses **Factory, Workload, Algorithm and Experiment**. The compiler
builds detached Scenario, Composition and Policy objects in memory and calls the
existing simulator and learners. You do not need to write those internal files
for a v4 experiment. Existing v2/v3 files remain explicit compatibility inputs.

This guide describes the configuration contract and lifecycle. Engineering
verification belongs in its separate validation record; example fragments below
do not constitute prepared research experiments or calibrated parameter choices.

## Ownership and directories

| Input | Schema | Owns |
| --- | --- | --- |
| Factory | `smartsom.factory/v2` | Geometry, resources, compatibility, machine speed/quality, capacities and optional reliability rules |
| Workload | `smartsom.workload/v3`, or existing v2 | External Job pool, specifications, arrivals, normal/rush attributes, due allowances, initial history and measured V |
| Algorithm | `smartsom.algorithm/v2` | Rules/models, Agent groups, observation/network contracts, learner/backend and reward transforms |
| Experiment | `smartsom.experiment-config/v4` | Task, selected inputs, independent seeds, budgets, validation/evaluation, runtime, execution, display and output |

Reusable author files belong in `configs/factories/`, `configs/workloads/`,
`configs/algorithms/` and `configs/runs/`. `configs/test/` holds engineering inputs.
Generated output belongs in the experiment's `output.root`; historical records
remain in `backup/`. A referenced file is shared input, not a file that execution
rewrites. Saving a new result does not update a Factory or Studio template.

References resolve relative to the file declaring them. Experiment input
references and `output.root` resolve relative to the Experiment; model references
inside Algorithm resolve relative to that Algorithm. CLI selectors resolve from
the current directory. Absolute references are also accepted.

The `schema` string identifies the file's typed format and validation rules. It
is not the experiment's name, model architecture or a command to launch training.

## Factory: layout, H and breakdowns

Existing v2 Factory files need no rewrite. Reliability is an optional top-level
field alongside `factory` and `authoring`; this fragment shows its shape:

```yaml
reliability:
  enabled: true
  defaults:
    enabled: true
    uptime:
      distribution: uniform
      min_ticks: 300
      max_ticks: 600
    repair:
      min_ticks: 10
      max_ticks: 30
  machines:
    machine_001:
      enabled: false
```

Replace the uptime object with `distribution: exponential` and `mean_ticks`
for exponential available-time sampling. Overrides use stable Machine IDs and
inherit unchanged default fields. Unknown IDs, reversed ranges or incomplete
active rules fail validation. A global `enabled: false` disables faults.

Uptime includes idle time while a machine is normally available. Repair does not
sample another failure. A failure pauses processing, retaining the Job, selected
mode and completed work; repair resumes the same operation. The run freezes the
realized half-open outage intervals. A Factory stores the sampling rules, never
the generated event schedule. Studio preserves these fields when reading and
saving; this change does not add a reliability editor panel.

H reports speed, quality and combined components from the declared machine table.
Breakdowns are an independent factor and do not enter H. For paired H conditions
of one scale, checks compare geometry/resource identities, operation compatibility
and nominal per-operation/per-mode capacity and capacity-weighted quality. A
legacy Factory without the required three-mode contract receives a diagnostic
rather than a fabricated H value. Nominal equality does not imply equal realized
throughput or eliminate duration-rounding differences.

## Workload: external data, due dates and V

V3 exposes these sections:

| Field | Meaning |
| --- | --- |
| `templates` | Each template's stable `id`, operation `route`, reference `times` in seconds, `count` per shared segment and `novel` history flag |
| `segments` | Number of identical quota segments; each segment contains whole arrival windows |
| `personalization` | Category weights for standard/parameter/insertion/mixed, adjustment magnitude bounds, insertion ratios and declared decimal precision |
| `arrivals` | Interval and initial batch, Jobs per window, physical seconds per tick and optional explicit sorted arrival slots |
| `due` | Base seconds, reference-work factor, seconds per operation, independent rush probability and shorter positive rush ratio |
| `history.counts` | Independent sample count per common template; novel templates cannot populate initial history |
| `volatility` | Selected `level`, history update `eta`, finite swap proposal budget and minimum separation |
| `occurrence_bound_seconds` | Optional declared support bound, checked before sampling; otherwise derived from generation bounds |
| `input_id`, `mode`, `tick_limit`, `horizon_multiplier` | Supply location and common finite/dynamic horizon settings |

Default category weights are `1, 2, 1, 1`. Counts must realize exact quotas.
Personalization modifies a uniformly selected nonempty subset of original
positions. Each selected position independently chooses increase/decrease and a
uniform magnitude; insertion samples uniformly among legal gap/type pairs.
Reference seconds are stored at the declared precision. Fractional reference
work remains exact through machine rate/mode scaling before physical rounding.

The normal allowance is

```text
base_seconds
+ reference_work_factor × final_reference_work_seconds
+ operation_seconds × final_operation_count
```

Rush multiplies that unrounded allowance by `rush_ratio`; the result is rounded
up once to physical ticks. Due time is external arrival plus the Job's frozen
allowance. Input waiting and replacement attempts do not reset it. Rush is a
public attribute after arrival, not an environmental rule forcing priority.
Reward weighting belongs to Algorithm, so a rush label does not itself introduce
a new late-penalty transform.

Content, rush, initial history and reference/Low/High ordering have separate named
random streams. One frozen pool contains Job identities, segment membership,
final specifications, personalization categories, rush labels and allowances.
Initial history is independently generated statistical reference, not initial
WIP or a sample drawn from future production Jobs.

Mid starts with a random reference permutation within each segment. Its window
rush counts become the common plan. Low and High independently propose exchanges
of same-class Jobs in the same segment but different windows. Low accepts only
strictly lower full-sequence V; High only strictly higher V. Each candidate is
compared from the identical initial history, with exact edit-cost transport and
compare-before-update history. The finite proposal budget includes rejected
proposals. The constructor requires `Low < Mid < High` and the configured minimum
separation; otherwise it reports failure without redrawing Jobs or rush labels.

All levels preserve the same pool, arrival slots, per-window rush counts and
individual allowances. V ignores Job IDs, rush, due dates, machines, queues,
replacement attempts and quality draws. Frozen provenance records actual window
curves, exact scores, reference/final permutations, edit rules, bounds, history,
quotas, realized rush fraction and stream identities. Future Jobs, whole-sequence
V labels and unrealized events are not Agent inputs. V3 rejects advance notice
that would expose a future Job before its external arrival.

**Generation is separate computation.** Exact transport and complete-sequence
scoring can be expensive for large pools. `check` does not start a learner or
worker, but it may generate and score datasets. Within one process, matching
config/data/split combinations reuse a bounded cache across maps, H and V; this
is not a persistent cross-command dataset cache. Frozen runs reuse saved data on
recovery. The 2000-Job/100-proposal generation time is not qualified by short
engineering cases. Reducing history, sampling approximate distances or skipping
separation changes the scientific contract; these are not silent performance
shortcuts.

Existing v2 explicit demands and profiles remain readable. V2 inputs do not gain
an inferred V label. Explicit demands are already frozen; v2 profiles are
materialized under the independent data stream.

## Algorithm: Agents and learner in one place

`mode: rules` requires exactly `machine`, `buffer`, `dispatcher` and `mover`
Agent roles, with rule policies and no learner:

```yaml
schema: smartsom.algorithm/v2
mode: rules
agents:
  machine:
    default: {kind: rule, name: spt}
  buffer:
    default: {kind: rule, name: edd}
  dispatcher:
    default: {kind: rule, name: nearest}
  mover:
    default: {kind: rule, name: shortest_path}
```

Each role can declare `group`, stable-ID `overrides` and a registered role reward
extension. An entity override contains an explicit `group` and `policy`. Policies
use `kind: rule` with a registered name/version/parameters, `kind: new_model` with
`extensions` and `projection`, or `kind: model` with `model.source`, checkpoint and
source group. A sharing group must have one compatible policy contract; interface
or network changes require an explicit independent group. Different roles do not
silently share one incompatible contract.

`mode: central` instead requires an exclusive controller and explicit PPO learner:

```yaml
schema: smartsom.algorithm/v2
mode: central
learner:
  backend: sb3
  algorithm: ppo
  gamma: 0.99
  parameters: {learning_rate: 0.0003}
controller:
  kind: new_model
```

Centralized PPO supports the existing SB3/RLlib backends. `mode: resource`
requires explicit RLlib and the four role declarations; existing resource PPO or
DQN can train selected model groups with rule/model partners. One training run has
one learner algorithm/backend. Central DQN, SB3 resource training and mixed
learners fail checks.

`learner.parameters` contains the existing typed PPO or DQN parameters, including
defaults even when omitted in YAML. Base/research reward transforms belong in
`learner.reward`; supported role transforms can be selected in Agent settings.
New-model `extensions.observation` is a name/version/parameter reference.
PPO `extensions.network` has `actor` and `critic` branches; DQN has `q`. Each branch
selects a registered encoder, hidden sizes and activation. `projection` owns
public projection limits and scales. Built-in defaults remain available.

These are general interfaces for research extensions. They do not implement a
Social Learning method, experience exchange, distillation or a top-level social
switch. See [student rules](student-rules.md) for the framework-free extension API.

## Experiment: task and one selected combination or matrix

An illustrative single Experiment in `configs/runs/` has this shape:

```yaml
schema: smartsom.experiment-config/v4
task: evaluate
factory: ../factories/large.yaml
workload: ../workloads/chosen.yaml
algorithm: ../algorithms/rules.yaml
seed: 101
data_seed: 0
evaluation:
  replications: 5
  record: true
output:
  root: ../../runs
  name: comparison
```

The referenced Workload and Algorithm must actually exist; this guide creates no
files. Training tasks additionally require an explicit `training` object with
`total_ticks`, `ticks_per_update`, `max_ticks` and optional `record_initial`.
`task` is `evaluate`, `train` or `train-evaluate`; it is not inferred from the
presence of training fields.

Use `matrix` instead of the single `factory`/`workload` fields:

```yaml
matrix:
  factories: [../factories/condition_a.yaml, ../factories/condition_b.yaml]
  workloads: [../workloads/low.yaml, ../workloads/high.yaml]
  seeds: [101, 102]  # training tasks only
```

The matrix expands Factory × Workload × training seeds, with one Algorithm.
Evaluation does not accept a training-seed axis. Rules use real evaluation
replications, not duplicate results relabeled as independent training seeds.
Repeated execution combinations are rejected.

`seed` controls policy/learning randomness. `data_seed` owns external data and
named train/validation/evaluation domains and replications. Changing the learner,
policy seed, map, H or faults does not draw a different external Workload.
Validation is in-training model selection; final evaluation is held-out execution.
V4 uses `data_seed` as the single author-controlled data root; the existing
validation/evaluation seed fields are not additional independent data roots.
Their physical cases and budgets are recorded separately from training progress.

`runtime` contains `device` (`cpu`/`cuda`), `num_envs`, `sampling_processes`,
`numerical_threads` and `environment` settings. Environment owns processing
variation, quality visibility, transport, rounding and optional horizon override.
New v4 execution defaults to `processing_rounding: ceil`; old v3 defaults and
scientific identities are unchanged.
V3 Workload owns its arrival mode. This interface does not add MPS, shared Ray
cluster management, automatic server discovery or GPU/cluster qualification.

`execution` owns orchestration: `executor`, `background`, `max_concurrent`,
`performance`, office/throughput `mode`, fixed/adaptive `scheduling` and
`calibration_seconds`. Defaults are foreground/native/one concurrent entry and
performance off. `logging` controls progress, summaries, debug, text/JSON and
optional integrations. `checkpointing`, `validation` and `evaluation` retain
supported existing options; v4 cases come from the selected Workload, so separate
scenario-file lists are rejected.

## Commands and temporary overrides

```sh
smartsom check experiment.yaml
smartsom run experiment.yaml
smartsom run experiment.yaml --factory factory.yaml --workload workload.yaml --algorithm algorithm.yaml
smartsom run experiment.yaml --set algorithm.learner.parameters.learning_rate=0.0001 --set experiment.training.total_ticks=4096
smartsom run experiment.yaml --background
smartsom monitor RUN_DIRECTORY
smartsom stop RUN_DIRECTORY
smartsom resume RUN_DIRECTORY --background
```

`check` and `run` share input compilation. Check reports expanded entries, actual
sources, H/V diagnostics, method groups/backend, budgets, cases, output and
execution mode. It creates no run directory and starts no workers, learners, Ray
runtime or performance calibration. It does not measure whether a laptop remains
responsive under load.

`--task` can override a new file's explicit task. `--seed` and `--data-seed` have
the ownership above. Selectors first replace inputs, then repeated `--set`
overrides apply. A selector narrows its matrix axis to the chosen file. Supported
namespaces are `factory`, `workload`, `algorithm` and `experiment`; fields resolve
against the validated models including defaults. Unknown fields, invalid types,
duplicate or overlapping overrides fail. Overrides apply to all selected matrix
entries, are frozen in provenance and never edit public files. Custom parameter
objects must declare the keys being overridden.

`--performance off|recommend|auto` uses the existing training calibration for
supported learning **train-evaluate** tasks. Recommend measures and returns its
recommendation without starting the requested full experiment; auto adopts the
measured recommendation and executes. These modes currently require the optional
Tune environment. Rule evaluation and unsupported task/learner combinations fail
explicitly; use native manual concurrency for rules. Calibration tunes supported
resource choices, not PPO/DQN scientific hyperparameters or network architecture.

Performance `recommend` or `auto` selects the Tune executor, and `check` reports
that effective executor. `off` disables calibration; `executor: tune` together
with `performance: off` is rejected. `execution.max_concurrent` sets native
worker concurrency. Tune chooses its concurrency from measured calibration;
non-default native concurrency and resume concurrency overrides are rejected
for Tune plans.

Old `run --config` and legacy training/evaluation entries retain their established
semantics and migration messages on stderr. Studio, playback and report/export
commands remain available. An existing saved run can be evaluated through the
source/checkpoint interface described in [command workflow](command-workflow.md).
For a completed single-entry native or Tune plan, `--source` accepts the outer
author run directory. A matrix requires selecting an individual training run.
Author source evaluation preserves the saved per-replication data and environment
randomness. Display, recording, deterministic policy execution and selecting a
smaller number of frozen replications are allowed; new cases, changed data seeds
or additional replications require a new Experiment.

## Background, stopping and recovery

Foreground remains default. Explicit background execution on macOS/Linux uses a
new process session, closes terminal stdin and writes separate stdout/stderr logs.
A launch reports success only after the actual driver registers its verified run
identity. It returns the run directory, state and log locations. Closing the
original terminal does not stop the detached driver. Successful launch is not
successful preparation, calibration or experiment completion; later failures are
saved in run state. Windows background launch reports unsupported.

Background logs contain durable summaries rather than an interactive ANSI panel.
`monitor` reads saved progress and uses the existing renderer; it does not keep the
experiment alive or change it. Native entry counts distinguish active, queued and
completed work. Evaluation does not invent training updates; validation does not
advance training ticks. Calibration and resource waiting are distinct stages.

`stop RUN_DIRECTORY` cooperatively stops new dispatch and waits for safe
boundaries. Its default timeout is 60 seconds; timeout is non-success and does not
implicitly kill processes. Explicit `--force` after timeout may terminate only
verified owned process identities. PID reuse, old unregistered runs, unrelated
applications and shared Ray runtimes never authorize guessing at ownership.
Stopped, failed, completed and forced-stop states remain distinct.

New v4 plans, including a single entry, keep a training/evaluation stage ledger.
Native entries store this ledger in `entries/<id>/stages.json`. Delegated Tune
entries retain their existing verified adaptive commit ledger under `performance/`;
completion requires an `experiment_complete` commit including final evaluation.
Resume skips completed stages. Interrupted training restores the latest complete
native update when its existing continuation requirements are met. Interrupted
evaluation restarts that stage, not the interior of a case. Resume permits only
supported execution/display changes; scientific inputs, code or extension drift
require a new run. Existing v2/v3 saved runs retain their older strict recovery
limits and are not migrated into this stage ledger.

The new plan directory contains:

| Path | Purpose |
| --- | --- |
| `plan.json` | Frozen `smartsom.author-plan/v1` inputs and entry plan |
| `batch.json`, `run.json` | Driver status/result records |
| `inputs/entry-NNNN/config/prepared.json` | Detached prepared scientific input for each entry |
| `entries/entry-NNNN/stages.json` | Training/evaluation completion, attempts and saved stage paths |
| `entries/entry-NNNN/runs/` | Actual child execution outputs |
| `logs/progress.json` | Saved progress consumed by monitor |
| `logs/stdout.log`, `logs/stderr.log` | Detached driver output when launched in background |
| `control/` | Runtime registration and stopping/launch control, separate from scientific inputs |

The local POSIX design does not constitute Linux, GPU/H20 or CARC acceptance.
Consult the validation record for actual platforms and checks.
