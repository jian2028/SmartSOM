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

For small engineering cases, the same v3 Workload schema also accepts a
nonempty `demands` list of fixed Jobs, or a simple seeded `profile`, instead of
`templates`. These three sources are mutually exclusive. Fixed Jobs preserve
their stable demand and operation IDs, per-machine nominal times, arrival and
reveal ticks, due dates, priority, and optional rush label. `mode` may be
`static` for fixed Jobs; paired-template generation remains finite or dynamic.
Only the paired-template form computes V. A fixed or simple-profile case must
not be labeled Low/Mid/High V. These forms all belong to Workload, not to a
separate author Scenario file.

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

New four-file experiments require Workload v3. Fixed demands and simple profiles
are represented within that schema; Workload v2 is rejected by the four-file
compiler. These engineering forms do not gain an inferred V label. Fixed Jobs
are already specified, while simple profiles materialize under the independent
data stream.

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

## Experiment: task and input matrix

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

Use `matrix` instead of the single `factory`/`workload` fields. An optional
`matrix.algorithms` list replaces the top-level `algorithm` scalar; specify
exactly one of these Algorithm forms:

```yaml
matrix:
  factories: [../factories/condition_a.yaml, ../factories/condition_b.yaml]
  workloads: [../workloads/low.yaml, ../workloads/high.yaml]
  algorithms: [../algorithms/base.yaml, ../algorithms/shaped.yaml]
  seeds: [101, 102]  # training tasks only
```

The matrix expands Factory × Workload × Algorithm × training seeds in that
order. Each entry reads and freezes one Algorithm, trains independently, and
keeps its own checkpoints and evaluation. A top-level single Algorithm remains
valid with a Factory/Workload matrix and preserves its previous expansion order
and scientific identity. All listed Algorithms must support the task; a rules
Algorithm in a training matrix fails `check` rather than being skipped.
Evaluation does not accept a training-seed axis. Rules use real evaluation
replications, not duplicate results relabeled as independent training seeds.
Repeated execution combinations are rejected.

`seed` controls policy/learning randomness. `data_seed` owns external data and
named train/validation/evaluation domains and replications. Changing the learner,
policy seed, map, H or faults does not draw a different external Workload.
Validation is in-training model selection; final evaluation is held-out execution.
V4 `task: train` freezes the held-out evaluation worlds in the saved run but
does not execute them. Later `smartsom evaluate RUN_DIRECTORY --checkpoint last`
uses those frozen worlds; a saved run with no frozen evaluation worlds fails
explicitly instead of reporting a successful zero-case evaluation.
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
`tuning`, balanced/performance `mode`, fixed/adaptive `scheduling` and
`calibration_level: online|off|quick|full`, optional `calibration_seconds`,
`calibration_candidate`, `preflight: quick|full` and
`preflight_coverage: each|representative`. New train-evaluate inputs without an
explicit executor/tuning selection default to online auto; train-only and
evaluation retain native execution. `logging` controls progress, summaries, debug, text/JSON and
optional integrations. `checkpointing`, `validation` and `evaluation` retain
supported existing options; v4 cases come from the selected Workload, so separate
scenario-file lists are rejected.

## Commands and temporary overrides

```sh
smartsom check experiment.yaml
smartsom run experiment.yaml
smartsom check path/to/experiment-directory
smartsom batch-run path/to/experiment-directory
smartsom batch-run path/to/experiment-directory --calibration-level full
smartsom run experiment.yaml --factory factory.yaml --workload workload.yaml --algorithm algorithm.yaml
smartsom run experiment.yaml --set algorithm.learner.parameters.learning_rate=0.0001 --set experiment.training.total_ticks=4096
smartsom run experiment.yaml --background
smartsom run experiment.yaml --tune auto --mode performance --calibration-timeout 20m
smartsom run experiment.yaml --preflight full --preflight-coverage each
smartsom attach RUN_DIRECTORY
smartsom monitor RUN_DIRECTORY
smartsom stop RUN_DIRECTORY
smartsom resume RUN_DIRECTORY --background
smartsom resume RUN_DIRECTORY --retry-failed
```

`check DIR` checks every direct `.yaml`/`.yml` V4 Experiment in that directory
without allocating a run. `batch-run DIR` freezes the same complete list and
runs it under one parent directory; `run FILE` remains a single-Experiment
command. In an interactive terminal, `batch-run` shows a preparation spinner
while it checks and freezes inputs, before the parent Rich monitor is available.
The parent monitor keeps the preflight, performance and formal-execution bars
visible across stages. Rule evaluations show completed cases and the active
case's tick progress; their file-level completion remains separate from that
case progress. An optional top-level `batch` block in each Experiment controls
stage order, concurrent non-training files, and a post-evaluation gate:

```yaml
batch:
  stage: 10
  parallel_files: 2
  gate:
    min_cases: 5
    min_deliveries_each: 1
    require_first_pickup: true
```

When any member declares `batch`, every member must declare it; otherwise
filename order is serial. Files in one stage must agree on `parallel_files`.
For `train-evaluate`, all files in a stage enter one Tune queue; measured
resource admission controls concurrent entries, so `parallel_files` does not
limit their training concurrency.
The gate applies only to evaluate tasks. It rejects incomplete/engineering
failed cases and can require delivery and pickup evidence; a later stage does
not start if it fails. These fields affect directory execution, not the
scientific identity of an entry. An edited source YAML cannot change a saved
batch on resume.
Directory `check` does not accept per-Experiment selectors or scientific
overrides; apply those in the individual Experiment files before freezing.

`train-evaluate` files in one stage share one frozen Tune batch and one
resource policy, even when the source Experiments specify native/off for
standalone execution. First, all expanded entries receive independent
128-tick no-update engineering smokes. Online execution skips isolated performance
calibration and adjusts concurrency from real updates. When optional calibration
is selected, `quick` has a five-minute budget for the whole batch and `full` has
30 minutes. An explicit
`--calibration-timeout 45m` takes precedence. Quick measures one disposable
512-tick case; full tries one representative formal-length case and an update
within the deadline. Compatible local history supplies a candidate, which is
always remeasured; `--calibration-candidate best` or a report path chooses a
different source. The current recommendation is bound to the batch plan and
reused on resume. A timeout without valid measurement retains the original
layout, marks it uncalibrated and ramps admission after a committed update.
An engineering failure blocks training. The training stage uses one ordered
queue across files, with global, file, group and live resource limits. Missing
mixed evidence permits conservative refill but marks schedule timing
uncalibrated; a measured slower pair cannot overlap. No probe measures final
held-out evaluation time or guarantees a global optimum.
No overall batch deadline is imposed.

`check` and `run` share input compilation. Check reports expanded entries, actual
sources, H/V diagnostics, method groups/backend, budgets, cases, output and
execution mode. It creates no run directory and starts no workers, learners, Ray
runtime or performance calibration. It does not measure whether a laptop remains
responsive under load.

`--task` can override a new file's explicit task. `--seed` and `--data-seed` have
the ownership above. Selectors first replace inputs, then repeated `--set`
overrides apply. A selector replaces its matrix axis with the chosen file for
this command, even if it was not listed in the YAML. `--algorithm` can also
replace a top-level single Algorithm. Supported
namespaces are `factory`, `workload`, `algorithm` and `experiment`; fields resolve
against the validated models including defaults. Unknown fields, invalid types,
duplicate or overlapping overrides fail. Overrides apply to all selected matrix
entries; an `algorithm.*` override must validate against every selected
Algorithm. Effective sources and overrides are frozen in provenance and never
edit public files. Custom parameter
objects must declare the keys being overridden.

Optional `validation.best_mode: completion_delivery_return` ranks each entry's
validation updates on the same frozen cases by completed-case count, mean
qualified deliveries over all cases, then mean raw return over all cases.
Exact ties retain the earlier update. An engineering exception or nonfinite
delivery/return makes that update ineligible; if no update is eligible,
`evaluation.checkpoint: best` fails rather than using `last`. This mode does not
compare or select winners across Algorithm or seed entries.

`--tune off|recommend|auto` uses the training calibration for
supported learning **train-evaluate** tasks. Recommend measures and returns its
recommendation without starting the requested full experiment; auto adopts the
selected execution mode and executes. New train-evaluate inputs without an
explicit executor/tuning selection default to auto with online feedback. These
modes require the optional Tune environment. Rule evaluation and unsupported task/learner combinations fail
explicitly; use native manual concurrency for rules. Calibration tunes supported
resource choices, not PPO/DQN scientific hyperparameters or network architecture.
The corresponding Experiment YAML is:

```yaml
execution:
  tuning: auto
  mode: performance
  calibration_level: online
  max_concurrent: 8  # Hard ceiling; omitted online ceilings use expanded entry count.
```

Online starts one real experiment, measures two stable windows of committed
physical ticks over common wall-clock time, then tests one more concurrent
experiment. A gain below 5% returns to the last accepted count with checkpoint-
safe pause/resume. Live memory peaks and CPU/RAM/GPU admission remain binding.
Validation and checkpoint time count toward useful throughput. Different task
groups are measured separately; this does not promise a globally optimal mix.
Threads, environment count and sampling processes stay fixed during online
training. A single-entry input cannot gain experiment concurrency.

Each run retains `online-performance.json`; compatible machine/task/source
records in `runs/.performance-profiles/online.json` shorten later confirmation
windows. Cached speed is advisory and does not skip live resource checks.
Resume restarts live observation conservatively while retaining prior evidence.

To run the optional complete test without the full experiment:

```sh
smartsom run experiment.yaml --tune recommend --calibration-level full
```

Without an explicit timeout, quick uses five minutes and full uses 30 minutes.
`off` runs no performance probes or historical-profile selection. It freezes
the declared runtime layout and concurrency as uncalibrated, then admits one
training entry until a committed update supplies a measured resource peak.
Online and `off` require a zero isolated calibration budget; neither skips
engineering preflight. `--calibration-timeout 10m` overrides quick/full and the
YAML budget for this
invocation. `--calibration-candidate latest|best|REPORT` selects a compatible
historical candidate for fresh measurement. The run stores its own report;
local history is only advisory.
Balanced leaves a larger office reserve; Performance leaves only the smaller
system reserve. Both select by measured batch throughput and can tie.

Tuning `recommend` or `auto` selects the Tune executor, and `check` reports
that effective executor. `off` disables calibration; `executor: tune` together
with `tuning: off` is rejected. `execution.max_concurrent` is a starting candidate
for tuning and sets native
worker concurrency. Tune chooses its concurrency from measured calibration;
non-default native concurrency and resume concurrency overrides are rejected
for Tune plans. In online mode, the declared concurrency is instead a hard ceiling.
Use `--tune off` for native manual execution; existing explicit native/off YAML
is not silently switched to online. Existing frozen runs keep their stored mode.

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

Interactive v4 `run` starts a background driver and attaches a controlling Rich
view by default. `d` detaches; pressing Ctrl+C twice within three seconds
requests a cooperative stop. The `p` key steps optional full-smoke coverage from
every entry to representative entries and then skip. Required preflight checks
always run. `smartsom attach RUN_DIRECTORY` restores this view; `monitor` is
read-only, and its double Ctrl+C closes only the monitor. `--background` returns
without attaching; `--no-background` retains foreground execution. Background
execution on macOS/Linux uses a new process session, closes terminal stdin and
writes separate stdout/stderr logs.
A launch reports success only after the actual driver registers its verified run
identity. It returns the run directory, state and log locations. Closing the
original terminal does not stop the detached driver. Successful launch is not
successful preparation, calibration or experiment completion; later failures are
saved in run state. Windows background launch reports unsupported.

Background logs contain durable summaries rather than an interactive ANSI panel.
`monitor` reads saved progress and uses the existing renderer; it does not keep the
experiment alive or change it. Native entry counts distinguish active, queued and
completed work. Evaluation does not invent training updates; validation does not
advance training ticks. The shared axis highlights preflight (purple), performance
calibration (amber), training (blue) and final evaluation (green). Each active
stage uses its own unit; there is no synthetic total percentage. `quick` preflight
checks every frozen input and policy contract. `full` adds bounded, disposable
policy smoke for every entry by default; it does not optimize weights or measure
throughput. Calibration and resource waiting remain distinct stages.

`stop RUN_DIRECTORY` cooperatively stops new dispatch and waits for safe
boundaries. Its default timeout is 60 seconds; timeout is non-success and does not
implicitly kill processes. Explicit `--force` after timeout may terminate only
verified owned process identities. PID reuse, old unregistered runs, unrelated
applications and shared Ray runtimes never authorize guessing at ownership.
Stopped, failed, completed and forced-stop states remain distinct.

For directory batches, use the **parent** run directory with `attach`,
`monitor`, `stop` and `resume`. The parent owns the Rich progress snapshot and
child runs live beneath `experiments/` and `performance/`. A completed file or
stage is verified and skipped on resume. This does not redo the shared
performance calibration. `resume PARENT --retry-failed` retries failed Tune
entries and retains completed ones. Closing an attached view leaves its
detached driver running.
During a Tune stage, the driver periodically rechecks its frozen implementation,
checkout and dependency identity. If they change, it fails the batch instead of
starting later workers against different source. A failed actor whose identity or
sampling-child ownership cannot be verified also fails the batch while retaining
its resource reservation until shutdown; later entries are not left waiting
indefinitely. Completed entries and their frozen evidence remain in the run.
Worker setup and update errors are saved in each attempt's
`logs/worker-error.json`; the parent batch and monitor show that cause when Ray
does not provide a trial exception. A confirmed stop can leave a child ledger
at its last pre-stop snapshot; the parent control state is authoritative.

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

Newly allocated v4 entries store detached `frozen-scenario`, `frozen-workload`,
`frozen-composition` and `frozen-policy` records and execution-config v2. These
schemas identify runtime snapshots; they are not additional files to author.

The local POSIX design does not constitute Linux, GPU/H20 or CARC acceptance.
Consult the validation record for actual platforms and checks.
