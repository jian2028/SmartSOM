# Current experiment workflow

SmartSOM uses Scenario, Algorithm and Experiment as human-authored inputs.
Scenario combines factory geometry, shared machine/AGV/Buffer defaults, job rules
and breakdowns. Studio edits and saves that same scene. The compiler expands it
into the existing physical contracts and freezes both source and resolved inputs.
See [scene authoring](scenario-authoring.md). Historical experiments use their
original frozen source; a format migration does not migrate checkpoints.

See [algorithm components](algorithm-components.md) for the current v2 Algorithm
schema, mixed role bindings, independent Mover and action ablations. Current definitions must explicitly select their components.

## Install and inspect

Use Python 3.12 and the committed uv lockfile. Optional packages stay outside core
imports. For native learning and both monitoring sinks:

```sh
uv sync --locked --extra learning --extra cpu --extra wandb --extra tensorboard
uv run --no-sync smartsom check current-smoke --profile local
```

The standalone resource catalog is [four named maps](../src/smartsom/config/maps/).
Current Scenario documents may embed their chosen map; the standalone-file limit
does not remove or split the Scenario authoring contract.

The two current factory scales use compact corridors for both Base and SL:

| Scale | Scene under `configs/scenarios/` | Machines / AGVs | Grid | Pre / post capacity per machine | Paired workload orders / initial |
| --- | --- | --- | --- | --- | --- |
| small | `small.yaml` | 8 / 8 | 22 × 6 | 4 / 4 | 128 / 16 |
| medium | `medium.yaml` | 20 / 20 | 27 × 12 | 4 / 4 | 320 / 40 |

Reference `configs/scenarios/small.yaml` or `medium.yaml` from an Experiment;
each includes its factory and paired job rules. Base and SL share these scenes.
The existing `current-*` examples use small. To inspect the medium scene
example, use `smartsom check experiments/examples/medium/default/experiment.yaml`. The medium scene includes its 320-order workload;
editing geometry alone does not implicitly change job rules.

Medium was formerly named target. Current `target*.yaml` files and the
`base_factory("target")` builder key are renamed to medium, including batch
examples. The selected compact resources now use factory IDs `small_factory` and
`medium_factory`; the wide resources preserve their original identities. The compact maps retain machine/AGV IDs, processing
modes, capacities and workload values, but change geometry and initial AGV
positions. A complete unit is Pre + Machine + Post (3 × 1 cells). Adjacent units
have two free cells horizontally and vertically; top/bottom boundaries and the
Input/Output columns have one free cell from the outer units. AGVs start above
their machines. Input/Output ports are one cell above the neighboring machine side ports
along the same boundary corridor. Each dock serves one facility and has one
physical vehicle occupancy slot.

Base coordinates use the actual grid width/height; Base and SL distances use
shortest paths on that same grid. The observation structure (`native-v2`) and
normalization scales remain unchanged. Generated due dates also depend on the
factory-specific nominal transport baseline, so some due dates change even with
the same workload parameters and seed. Shrinking distances changes traffic,
observations and potentially results: this is a new scientific condition, not
an equivalent relabeling of the historical maps. Existing calibration files and
checkpoints must not be reused as if their factory identity matched.

The pre-compaction maps (Small 24 × 11, Medium 29 × 19) remain available at
commit `1c071ad300d72e9f4ad33b065f05606c161d3df2` under the same factory paths.
Historical source and run artifacts remain in place; resume them with their
matching frozen inputs/source/environment. The test-only `historical_base_factory()` constructor in
`tests/helpers/historical_geometry.py` retains the original native-v2
physical trace regression; current corridor and observation tests exercise the
compact maps separately.

Before renaming, the small file matched the old Mac Base frozen source and its
manifest SHA256 `57af8dbc9910de83abe488f53276949c9049e6c9f00cf2fae51f40fb8a1dc5dd`.
The former target file matched the BudgetVM batch
`sl-nine-arm-20261027-a39664c` frozen source and manifest SHA256
`4fbc910f666abf34534be36d9f6e1d5937c56d6d9eac70697cae5ceec5ce9ab7`.
These hashes identify the historical files, not the compact maps.

Add `--extra tuning` for Ray Tune and `--extra studio` for the desktop editor and
recorded playback. `check` never opens W&B or allocates an experiment directory.
By default it shows a compact human-readable summary: task count, shared settings,
method differences, seeds, budgets, monitoring and output location. Large batches
show at most eight task rows; mixed settings are marked rather than represented
by the first task. This is configuration validation, not a resource or W&B login test.

```sh
smartsom check current-train --profile local
smartsom check current-train --profile local --verbose
smartsom check current-train --profile local --json
```

`--verbose` includes resolved files, component contracts, network ownership,
parameters, overrides and setting sources. `--json` preserves the full structured
check payload for scripts; scripts that previously parsed default stdout must add
this flag. The two options are mutually exclusive. Configuration errors retain a
nonzero exit code; `--json` also formats configuration errors as JSON on stderr.
The Python check API is unchanged. Run/resume display options are unchanged.

## Launch choices

```sh
uv run --no-sync smartsom run current-smoke
uv run --no-sync smartsom run current-train --profile server --background
uv run --no-sync smartsom run path/to/experiment-directory --max-concurrent 4 --concurrency-mode adaptive
uv run --no-sync smartsom run current-train --wandb offline --tensorboard
```

These are alternatives, not commands to run repeatedly into the same output path.
Training and evaluation require a completed calibration record. The bounded
engineering examples form this explicit sequence:

```sh
uv run --no-sync smartsom run current-calibrate
uv run --no-sync smartsom run current-train
uv run --no-sync smartsom run current-evaluate
```

The example budgets are engineering checks, not research recommendations. Change
the Scenario, Algorithm and Experiment files for formal experiments and launch those only from integrated main
or an explicit source tag. Existing output directories are never overwritten.

An existing path wins over a short name. Short names search `experiments/` by
file stem or directory name; ambiguity fails with candidate paths. Directory runs
read direct Experiment YAML members and preserve their declared stage order.
A directory containing
calibration/train/evaluate templates does not automatically resolve dependencies;
use a deliberately authored batch with available calibration inputs.

Profiles live in `configs/profiles/` or an explicit path. They are optional and do
not inherit other profiles. Priority is defaults, Experiment operation defaults,
explicit profile, explicit CLI. For example:

```yaml
launch:
  mode: foreground
concurrency:
  mode: adaptive
  max_concurrent: 4
resources:
  cpus: 8
  memory_gib: 12
tracking:
  mode: online
  project: SmartSOM
  tensorboard: false
```

Both supplied profiles default to foreground. `--foreground` and `--background`
are mutually exclusive. Profiles cannot change seed, learning rate, observation,
reward, train/validation budgets, sampling process count or device. Advanced
`--set algorithm.production.policies.ppo.learner.params.learning_rate=...` remains available for a new experiment and
is recorded explicitly. Resume uses frozen scientific inputs, never edited author
YAML. Concurrency limits independent trials; it does not change rollout size or
learning updates. Resource budgets are admission ceilings, not OS hard quotas.

For native train/evaluate/calibrate with `tune: off`, an optional
`resources.guard` adds cooperative worker protection independently of those
budgets. It requires all three fields: `max_worker_rss_gib`,
`min_host_available_gib`, and `min_disk_free_gib`; null (the default) disables it.
The rule-DQN recipe uses 8/8/2 GiB. Rollouts sample at most once per second;
model/update/checkpoint/receipt boundaries force checks. A breach records
`resource-guard-failure.json`, preserves committed results and requires explicit
recovery. It cannot enforce a hard peak during a single allocation or save.
See [ADR0069](decisions/0069-cooperative-worker-resource-guard.md).

Native PPO/DQN use Torch eval mode for rollout, bootstrap, optimizer forwards and
evaluation. Optimizer forwards still enable gradients and update trainable weights.
This keeps PPO's recorded and recomputed probabilities on the same distribution
and prevents DQN target inference from changing state. Dropout is disabled and
BatchNorm running statistics stay fixed; DQN target refresh copies the complete
state, including buffers. Custom networks must support singleton requests and
deterministic, batch-independent eval forwards. BatchNorm without running
statistics (`track_running_stats=False`) does not meet that contract. Action
sampling and DQN epsilon exploration remain explicit seeded policy operations.

## Find, control and compare

```sh
smartsom runs
smartsom status RUN_ID
smartsom status RUN_ID
smartsom attach RUN_ID
smartsom pause RUN_ID
smartsom resume RUN_ID
smartsom stop RUN_ID
smartsom clean RUN_ID --show-target
smartsom clean RUN_ID --confirm-target TOKEN_FROM_PREVIEW
smartsom results runs --algorithm ppo --seed 101 --status completed
smartsom config diff current-train RUN_ID
```

Commands accept a run directory or unique ID prefix. Supply `--root PATH` for
custom output roots; multiple roots can be provided. No mutation selects an
implicit `latest`. Historical runs are discoverable and readable; control or
continuation requires archived source. Pause requests and completed safe-boundary
pauses are separate states. A live scheduler resume releases its pause; a stopped
driver resume verifies frozen source/environment/inputs and committed state.
Weights alone are not full continuation state.

Interactive foreground CLI runs and `attach` share the same terminal controls.
Press `q` or `Ctrl+C` to open the exit menu: `1` / Enter safely stops and retains
all local results (the default); `2` safely stops then deletes all local results;
Esc or Ctrl+C cancels the menu. The target directory and deletion scope are shown
before choosing. Deletion uses the same ownership checks as the run cleanup backend,
waits up to 120 seconds without force, and preserves files on timeout. `d` only
detaches the interface; execution continues and can be reattached by run ID.
Normal completion does not delete results or prompt for deletion. Without an
interactive input/output terminal, commands retain their existing noninteractive
execution behavior. W&B startup failures in the foreground are relayed to the
terminal: only an explicit `y` authorizes offline continuation. Detaching or
terminal loss does not imply that consent.

`clean RUN_ID --show-target` previews whole-run deletion; use the returned
`--confirm-target` only after stopping and verifying process exit. `clean` refuses
a live run and never stops it implicitly. It permanently removes the local result
directory, including checkpoints, frozen copies, reports and offline tracking;
original authoring inputs and W&B cloud runs remain. To clean one removed task,
use `clean RUN_ID --task-id ID --show-target` and its target-bound confirmation.
`remove` only removes a task from scheduling and retains files.

`resume` releases a pause or restarts a stopped matching driver. It never opts into
failed-task retry. Use `retry RUN_ID --all` for a batch, or `retry RUN_ID` for a
failed single run. A live batch supports `retry RUN_ID --task-id ID`; its checkpoint
and ownership are revalidated by the coordinator. Paused tasks use `resume`, not
`retry`. Exact task retry after a driver exits is rejected: whole-batch recovery
uses `retry --all`. Completed work is retained. Add `--preview` to inspect without
submitting. Task pause/stop holds remain authoritative.

`status` combines run metadata, operational settings and the runtime journal.
Use `--task-id ID` for a task, `--command-id ID` for a control receipt, and `--json`
for complete machine-readable output. `runs` lists discoverable runs. `monitor`
is read-only; `attach` enables interactive controls.

A successful whole-run `stop` returns exit code 0 when owned processes have exited,
even if their final scientific status was failed or stopped. Timeout or rejected
controls return nonzero. Other control commands return 0 when their request is
accepted; `received` and `applied` do not certify that workers finished draining.

### Atomic runtime configuration

```sh
smartsom configure RUN_ID --max-concurrent 8 --concurrency-mode manual \
  --evaluation-processes 5 --evaluation-cpus 5 --cpus 30 --memory-gib 80
smartsom configure RUN_ID --task-id EXACT_TASK_ID --evaluation-processes 2 --evaluation-cpus 2
```

All supplied fields form one journal command/revision and validate before any are
changed. Missing fields preserve prior values. `--cpus`/`--memory-gib` constrain the
whole queue, while `--evaluation-cpus` constrains all evaluation workers within one
experiment. Eight experiments at five workers can request forty CPUs, but an
aggregate thirty-CPU budget limits actual parallelism. The cap is cooperative
admission, not OS isolation. Explicit task evaluation fields override queue defaults;
other evaluation fields inherit those defaults. Task scope rejects queue budgets
and experiment concurrency settings.

`run`, `check`, and stopped-driver `resume` use the same resource option names.
Profiles store `concurrency.evaluation_cpus`; null means no extra cap beyond the
process target. Live `configure` requires a supporting batch driver; single-run
settings change through stopped-driver `resume`. Old source identities are not
upgraded in place.

## Monitor and sync

W&B defaults to online/project `SmartSOM`; configure entity or use the account
default. `--wandb disabled` keeps only local events; `--wandb offline` also keeps
uploadable SDK records. Missing enabled dependencies fail before launch. Online
initialization failure requires an explicit interactive offline choice; background
launches must have offline selected in advance. Mid-run sync failures do not
invalidate local checkpoints. Training and synchronization have separate status.
If `log()` rejects an event before the SDK accepts it, `upload_errors` and the last
failed event sequence remain in tracking status across polling, shutdown and
continuation. A later successful SDK queue flush cannot prove that event was
uploaded. The complete event remains in the local journal; SDK offline sync does
not automatically replay rejected journal events.

Local `tracking/events.jsonl`, manifest, frozen inputs, progress, cases and
continuation checkpoints are authoritative. W&B URLs in status and terminal
output can be opened on a phone. The logger does not upload source or checkpoints
by default. Online continuation pins run ID/project/entity. Each offline segment
has its own ID and sync directory, with the shared experiment ID in its config.
Use each recorded `sync_command` from `tracking/status.json` to upload its segment;
do not assume offline segments merge into one online history.

`--tensorboard` writes events only. Start a localhost viewer separately:

```sh
smartsom tensorboard RUN_ID --port 6006
```

Shell completion is opt-in; SmartSOM does not edit shell startup files:

```sh
# Bash
source <(smartsom completion bash)
# Zsh (after compinit)
source <(smartsom completion zsh)
```

Historical sources and readers are referenced by `docs/archive/index.json`.
Current playback reads current recordings without executing policies; historical
recordings require the archived reader.

## Scientific matrices

An Experiment may declare explicit scientific override axes. Each combination
becomes one independent frozen experiment, scheduled under the same total
concurrency limit as all other files in the directory:

```yaml
matrix:
  experiment.run.seed: [101, 102, 103]
  algorithm.decision_protocol.pre.accept.allow_stop: [false, true]
```

Matrix axes use the same dotted scientific fields as advanced `--set`. Operational
axes are rejected. Explicit CLI science overrides apply after matrix values;
if they collapse distinct entries into duplicate scientific identities, compilation
rejects the batch. The matrix does not create a second nested worker pool.

Resource budgets are logical admission limits, not OS enforcement. Standalone
native runs check one CPU and observed/floor memory before execution; directory
and Tune schedulers additionally account for active workers and host headroom.
Tune performance calibration is explicit; adaptive runtime control preserves the
selected numerical-thread layout and changes independent experiment concurrency.

## Tracking labels and research views

`tracking.tags` accepts a list of free-purpose labels; `tracking.variant` optionally
names a research variant. The resolved algorithm, observation, factory scale and
control mode add automatic labels. Names use experiment, map, method, variant and
training seed (when one exists). Display labels do not change scientific identity.
Each launch supplies a unique group, and continuation retains the recorded group.
`tracking.upload_interval` defaults to five wall seconds: the local event journal
remains complete, validation summaries and phase/final events publish immediately,
and pending training values flush on close.

Install the optional dashboard extra and explicitly create a **new** saved view:

```sh
uv sync --locked --extra dashboard
uv run smartsom wandb dashboard RUN_OR_BATCH
```

This command creates private chart presets and new W&B saved views; it never loads
or overwrites an existing workspace. It uses the run catalog resolver and accepts
`--root` for explicitly selected search roots. Runs need recorded online identities;
offline segments must be synchronized and their cloud identities established before
using the dashboard command.

The six headline metrics have separate charts. Map, workload/protocol and selection
mode are separated before plotting. Method legends can toggle series; condition
and world controls operate on table rows, not on run-level config filters.
Per-seed summaries are uploaded as incremental Tables. Vega's `stdev` computes
sample standard deviation after deduplicating seed identities; one-seed intervals
stay missing. Provisional and complete seed coverage appear separately. Rule's
repeated references are not counted as independent training seeds.

Entity cumulative tables cover recorded evaluation cases. Cloud entity time series
are limited to the representative first world/repeat of each method, seed and
selection; `tracking/status.json` records the actual chosen case identities.
Detailed local evidence remains complete. Gray entity curves show a resource-class
aggregate; entity legends enable individual overlays. Utilization uses accumulated
numerators and denominators, not an unweighted mean of ratios. The companion local
statistics snapshot is explicitly labeled as a snapshot; the metric charts continue
to consume newly logged tables. Mobile layout is supplied by W&B itself.

## Incremental authored seed selection

A directory can contain `medium-base.yaml`, `medium-sl.yaml`, `small-base.yaml`
and `small-sl.yaml`, each with schema `smartsom.base-experiment/v1`. Scenario and
Algorithm YAML files are not executable entries. Existing canonical
`variant/experiment.yaml` layouts remain supported.

Keep the full scientific lists in each Experiment, for example
`train_seeds: [4101, 4102, 4103]`. Select execution separately:

```sh
uv run smartsom run experiments/my-batch --id medium-base --id medium-sl --first-train-seeds 1 --output-root runs/my-batch --background
uv run smartsom run experiments/my-batch --id medium-base --id medium-sl --resume-missing --output-root runs/my-batch --background
```

The first invocation freezes all 12 task inputs and starts only the two medium
4101 tasks. After that process exits, the second invocation starts the four
remaining medium seeds in the same root. Repeating the second command schedules
zero new tasks. Small tasks stay deferred. Preallocated directories are not trained
models. A completed selection with deferred tasks reports `partial`.

`--set experiment.train_seeds=1` still changes science to seed `[101]`; it does
not select the first element of the authored list. Resume rejects changed source,
maps, algorithms, case panels, full scientific configuration or directory binding.
Choose a new output root for a changed study. Existing batches created without a
full incremental inventory cannot be converted in place.

A live driver rejects a competing submission. Manual task holds survive resumption.
Failed tasks are not automatically retried: inspect the error and use explicit
`retry --all` for the same frozen batch when a complete continuation
exists. Training-complete tasks with pending final evaluation continue through
the native runtime. Earlier stages must complete before a selected later stage.


Append one named file or selected directory entries to an open queue:

```sh
uv run smartsom add RUN_ID experiments/other-batch/medium-base.yaml --first-train-seeds 1
uv run smartsom add RUN_ID experiments/other-batch --id medium-base
```

Both forms identify the same owning batch and seed directories. The second adds
only its missing seeds. Results remain in the owning batch's `trials/ENTRY/seed-N`
paths, not `QUEUE/added`. Cross-batch execution uses the queue's admission ceiling;
its ownership claim blocks a competing original-batch driver until the selected
work drains. Completion is reflected in the original batch, so a later
`run --resume-missing` does not run those tasks again. Only independent
stage-0 tasks may be appended. Closed queues require explicit resumption.

For a batch previously placed at an arbitrary CLI output root, provide that root
on its first submission to another queue, for example
`add RUN_ID path/to/batch/medium-base.yaml --output-root runs/custom-batch`.
The queue then remembers this batch's frozen owner root, so subsequent file,
directory or parent-directory additions reuse it. Multiple known destinations
require an explicit choice. There is no global search for unknown historical
CLI destinations: keep the same explicit root/profile when selecting an existing
study. `--output-root` with `add` selects one owning batch, not a parent
folder containing several independent batches.

For native evaluation panels, `--evaluation-processes 4` runs up to four frozen
cases concurrently; the equivalent Experiment/profile setting is:

```yaml
concurrency:
  evaluation_processes: 4
```

The default is `1`. This is separate from `max_concurrent`, which limits whole
trials. Native batch training retains a baseline CPU; validation/final evaluation
requests additional phase resources only while needed. Use an
adequate `resources.cpus`/`resources.memory_gib` budget. The current process mode
supports built-in components with `tune: off` and recovery disabled. Results and
tracking remain ordered; existing case receipts are reused when resuming.
