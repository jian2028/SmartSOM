# Unified command workflow

The main commands are `check`, `run`, `stop`, `resume`, `attach` and `monitor`.
In an interactive terminal, a v4 `run` launches a verified background driver
and attaches its Rich view. `d` detaches; two Ctrl+C presses within three
seconds request a cooperative stop. Explicit `--background` returns without
attaching, and `--no-background` runs in the foreground. A detached driver
continues after the terminal closes. `monitor` is read-only; `attach` restores
the controlling view.

The top of the Rich view has three long bars: preflight, performance
calibration, and training plus final evaluation. The current phase is
highlighted; phase names, status labels, bars and percentages use fixed aligned
columns. Experiment cards underneath show each training, periodic
validation, and evaluation case separately. The formal overview gives each
parallel experiment equal weight, then gives its training and final evaluation
equal phase weight after normalizing their own budgets. Calibration progress
during a run is elapsed wall-clock budget; an early completed calibration
displays its actual elapsed time and measured-candidate count. A separate
current-stage box shows the relevant counts and resource facts. Training cards
appear only after calibration; they stack at ordinary terminal widths and use
two columns when space permits. Compact views preserve the current case and
report hidden trials. Each card shows an approximate ETA after enough of that
experiment's own work has completed; otherwise it says `估算中`. The formal-stage
box reports the worker-confirmed CPU/GPU device, running experiment count,
environments per experiment, sampling processes and compute threads. Before
the first worker reports, it labels the configuration as pending rather than
presenting the calibration recommendation as an active allocation.

```sh
smartsom check experiment.yaml
smartsom run experiment.yaml --preflight full --preflight-coverage each
smartsom attach RUN_DIRECTORY
smartsom monitor RUN_DIRECTORY
smartsom resume RUN_DIRECTORY --background
```

See [four-file authoring](four-file-workflow.md) for schema ownership, selectors,
data seeds, named overrides, native matrices and performance scheduling. The
remaining examples describe compatible v3/Study/Tune entry formats.

```sh
smartsom check --task train-evaluate --config CONFIG.yaml
smartsom run --task train-evaluate --config CONFIG.yaml
smartsom stop RUN_DIRECTORY
smartsom resume RUN_DIRECTORY
smartsom monitor RUN_DIRECTORY
```

## Tasks and inputs

| Input | Tasks | Behavior |
| --- | --- | --- |
| `--config` experiment-config/v3 | train, evaluate, train-evaluate | Existing composable configuration graph |
| `--config` tune-batch/v1 | train-evaluate | Current preflight, calibration and Ray scheduling |
| `--source` saved v3 training run | evaluate | Frozen partners plus selected checkpoint |
| `--study` prepared native Study directory | train-evaluate | Existing prepared Study, without re-preparation |

Choose exactly one input. `train` includes configured validation but has no final
evaluation. `train-evaluate` includes both. Selecting `evaluate` never trains a
fresh network merely because the configuration contains training settings.

For legacy inputs, scientific settings, maps, workloads, scenarios, algorithms and
training budgets remain in their existing YAML files. New v4 authoring compiles
the four author files into the same simulation and learning execution boundaries.
Display settings include `--verbose`, `--no-verbose`, `--debug`, `--progress`,
`--log-format`, `--summary-interval` and `--progress-title`.
For single evaluation, `--seed` changes the evaluation seed; `--replications`,
`--scenario`, `--deterministic`, `--record`, `--replay`, `--render-mode human`,
`--render-case` and `--render-replication` select applicable evaluation behavior.
`--output-root` selects a new output location. V3 model initialization belongs in
policy model selectors; `--initialize-from` is rejected with that guidance.
Tune `--mode balanced|performance` and `--execution fixed|adaptive` apply only to a
Tune batch. A prepared Study uses its existing scheduling configuration.

```sh
smartsom check --task evaluate --source RUN_DIRECTORY --checkpoint best
smartsom run --task evaluate --source RUN_DIRECTORY --checkpoint best --preview
smartsom run --task train-evaluate --config BATCH.yaml --mode balanced
smartsom run --task train-evaluate --study PREPARED_DIRECTORY
```

Preview selects the first case and one repetition, records by default and disables
the additional full replay audit. It creates a separate run marked `purpose:
preview`; it does not change its source. Conflicting case, repetition and replay
overrides are rejected. Preview is not a complete evaluation result.

`check` resolves the selected task, model/input references, dependencies,
checkpoint conditions and output parent. It does not construct a learner, start
Ray, sample, calibrate or train. Passing these checks does not guarantee enough
live resources or successful production completion.

## Stopping and recovery

```sh
smartsom stop RUN_DIRECTORY --timeout 60
smartsom stop RUN_DIRECTORY --timeout 60 --force
```

New executions store separate `control/` metadata with their driver identity,
kernel process start identity and observed owned descendants. Ordinary stop writes
a cooperative request: no process signal is sent. Training completes a native
update, including validation and recovery saving, before stopping. Study stops
dispatching new entries and waits for active workers to reach their boundaries.
Tune stops admission and stops active trials after committed results; calibration
uses its cancellation path. Evaluation can stop at a physical tick boundary and
retains a partial recording; it does not resume inside that case.

The default wait is 60 seconds. A timeout returns a non-success exit status and
reports remaining processes. Only explicit `--force` permits TERM followed by
KILL for verified registered identities; unfinished work may be lost. Neither
ordinary nor forced stop kills by process name or shuts down a shared Ray cluster.
Batch-owned child directories point to the batch driver; stop the top-level
directory, not a child managed by that driver.

Legacy runs without registration, relocated control records, missing ownership
and reused PIDs are refused. Existing active runs are not retrofitted.
Monitor reads control state and reports stopped only after owned processes exit.
Process control currently targets macOS/Linux POSIX; platform qualification is
reported separately from implementation.

`resume` routes saved single, Study and Tune directories to their native restorers.
`--retry-failed` is available for Study/Tune only. Ordinary single-run resume
continues training from its complete recovery state, not arbitrary `best` weights
and not an interrupted final evaluation. Study and Tune retain their own frozen
identity and stage restrictions; an implementation change can make an older run
ineligible for strict resume. Stopping does not loosen those restrictions.

## Compatibility and auxiliary tools

Old `train`, `train-evaluate`, `evaluate`, `study`, `tune` and task-less `run` retain
their original semantics and print migration guidance to stderr. V2 presets and
recipes still use those compatibility paths. Structured logging retains JSON
stderr; JSON result stdout is unchanged. `tune recommend` remains available for
recommendation-only calibration, and `tune --study` imports a prepared Study into
a **new** Tune batch, unlike native `run --study`.

Studio, playback, show-config, authoring, reports, audits and exports keep their
existing responsibilities. None of these command changes introduces background
job management, new scientific configurations or a new simulation engine.
