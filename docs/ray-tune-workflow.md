> Historical workflow record. Current commands and supported configuration are documented in [current-workflow.md](current-workflow.md) and ADR0058; reproduce older examples using their archived source.

# Ray Tune batch execution

This is an execution tuner for frozen v3 experiments. It uses Ray Tune 2.58.0,
FIFO scheduling and `ResourceChangingScheduler`; it does not search scientific
hyperparameters or terminate experiments because their rewards are low.

New V4 train-evaluate inputs default to online concurrency feedback instead of
the isolated calibration described below. Use
`smartsom run experiment.yaml --tune auto --calibration-level online` to select
this explicitly, or `--tune recommend --calibration-level full` for an optional
complete test. Online keeps the authored threads and sampler fixed, compares real
committed throughput windows, increases concurrency one at a time and backs off
when the gain is below 5%. Its retained `online-performance.json` and separate
machine/task cache contain measured operating points, not a hardware-optimum
guarantee. Explicit native/off inputs and existing frozen runs retain their mode.
See [ADR 0040](decisions/0040-online-experiment-concurrency.md) and the
[four-file workflow](four-file-workflow.md) for defaults and resource ceilings.

Comparable initialized learner packages can share an online performance group
across seeds. Grouping verifies the effective network/encoder contract and keeps
reward, optimizer, sampler, workload shape and validation/save schedules distinct.
Initial learner weight bytes and seed provenance are excluded only from this
advisory performance key. Frozen partners and unverified historical model
contracts retain their fingerprints; scientific and continuation identities
still include actual inputs and payloads. See
[ADR0051](decisions/0051-online-learner-shapes-and-stable-monitor.md).

For a new directory batch, explicitly select online adaptive execution in its
Experiment YAML (`tuning: auto`, `scheduling: adaptive`,
`calibration_level: online`) and use `smartsom batch-run DIR --background`.
Its authored `max_concurrent` is a ceiling. Compatible seeds start at one and
grow only after current RSS and stable throughput observations. Incomparable
groups stay separate; a singleton group has no second experiment to admit.

A frozen Tune batch retains its saved scheduling mode and ceiling. `batch-limit`
controls an explicit live queue; it does not hot-edit an ordinary Tune batch.
Tune resume has no concurrency override. The local Batch05 PPO launcher uses
fixed/off with `SMARTSOM_FIXED_ADMISSION_SLOTS=4`, which must equal its frozen
global limit. That bounded startup reservation is not adaptive calibration and
cannot turn a running four-slot campaign into six slots. Use a new explicitly
configured run when a different frozen execution contract is needed.

Activate an environment that already has the optional learning and tuning
dependencies installed:

```bash
source .venv/bin/activate
```

For an allocated CUDA node, use `--extra cuda` instead of `--extra cpu`. CUDA
visibility and Slurm/cgroup/affinity limits must describe the current allocation.
The runner owns one local Ray runtime; do not run it inside another initialized
Ray runtime. Multi-node execution and Apple MPS are outside this adapter.

Create a batch file referencing your existing v3 experiment recipes. Paths are
relative to the batch file. This example references repository engineering
recipes; it does not create a daily Small experiment:

```yaml
schema: smartsom.tune-batch/v1
mode: balanced
execution: adaptive
active_limit: 1200
output_root: runs
entries:
  - id: machine-ppo
    config: configs/test/runs/train_machine_ppo.yaml
  - id: machine-dqn
    config: configs/test/runs/train_machine_dqn.yaml
```

```bash
source .venv/bin/activate
smartsom tune check --batch batch.yaml
smartsom tune recommend --batch batch.yaml --mode balanced --calibration-timeout 10m
smartsom tune run --batch batch.yaml --mode performance --calibration-timeout 20m
smartsom tune run --batch batch.yaml --preflight full --preflight-coverage each
smartsom attach runs/<tune-batch>
smartsom stop runs/<tune-batch>
smartsom resume runs/<tune-batch> --retry-failed
```

`check` validates every recipe, required backend, frozen device and writable
output parent without constructing a learner or starting Ray. `recommend` creates
an isolated batch and runs real calibration, stopping before formal batch
training. `run` performs the same checks and calibration and automatically uses
its measured recommendation. `resume` keeps stable experiment identities, skips
verified completed experiments, and creates a new Tune segment from each last
verified native commit. Failed experiments require `--retry-failed`.

A prepared v3 study can be imported with `--study <directory>` instead of
`--batch`. This checks the study plan and child snapshot hashes and copies inputs
and frozen partner models into a **new** batch. It does not resume or modify that
study or adopt its historical checkpoints. The new batch records its own source
identity and the imported plan's provenance. Its paired initial/rule/random
controls use the original frozen evaluation cases.

This import uses the compatibility `tune` command. Unified `run --study` instead
executes the prepared native Study directly. See the
[command workflow](command-workflow.md) for task/input combinations and safe stop.

Execution follows: preflight → resource observation → baseline for every workload
group → candidate measurements → repeated leader measurements → recommendation →
queued experiments → complete update checkpoints → final evaluation and controls.
Calibration groups retain algorithm/backend, learner shape, factory, device and
update quantum; each group uses a representative frozen workload. The requested
starting layout is tested as a baseline, while compatible experiments may share
the resulting measured layout. Measurements
include cold initialization, actual sampling/learning, validation and checkpoint
cost, concurrent process-tree RSS peaks and aggregate physical-tick throughput.
Calibration evidence and checkpoints are separated from experiment evidence and
are never used to initialize the formal learner.

Calibration searches experiment concurrency, independent environments, persistent
sampling processes and learner numerical threads. Candidate CPU requests include
one core per sampling child, and observed memory peaks include their process trees.
The chosen sampling layout is frozen before formal training and restored unchanged;
adaptive execution thereafter changes only learner threads and experiment
concurrency. Models, seeds, workloads, budgets, scientific parameters and CPU/CUDA
cohort remain frozen. CUDA calibration
currently measures one complete GPU trial at a time; multi-GPU performance and
H20/CARC hardware acceptance remain unverified.

`balanced` reserves at least one CPU and 20% CPU, and at least 2 GiB and 20% RAM.
`performance` reserves at least one CPU and 5% CPU, and at least 1 GiB and 10% RAM.
CPU attribution uses a matching observation window. For a process verified to
have been born since the preceding aggregate sample, its cumulative user/system
CPU counters measure that window even before `cpu_percent` has a second sample.
This prevents a new owned actor's measured startup work from being counted as
external pressure. Older newly observed processes remain unprimed, and new
external-process CPU is never deducted as owned load. The reserves and genuine
CPU/RAM pressure guards remain unchanged.
An owned process whose first counter cannot be reconstructed remains explicitly
unprimed. During that observation, extra admission and CPU resizing wait, and
the conservative external-load upper bound cannot establish CPU-only preemption
or update the external-load EMA. Measured RAM pressure still applies immediately;
normal CPU pressure decisions resume after the process counter is primed.
Memory admission uses observed peaks with a 1.25 factor. These are conservative
resource budgets, not operating-system CPU quotas or a measurement of GUI frame
rate. The calibration timeout includes resource waiting, probe startup, measurement
and cleanup. After a probe exits, calibration discards the CPU delta window
containing that probe and waits one normal sampling interval before the next
admission or final ranking. This wait counts against the same timeout. Probe
and confirmation budgets reserve that interval and the observed scan costs in
advance; actual deadline checks still reject any late sample. If no
fresh window fits before the deadline, calibration cannot recommend a profile
from the stale load sample; reserves and external-pressure checks are unchanged.
Stable leaders may end the search early. At the deadline, an
unconfirmed leader can still be the fastest valid measured candidate, marked
not converged. If none is valid, automatic execution attempts the user-specified
starting layout or the minimal layout and marks it uncalibrated; normal resource
admission and preflight can still reject the start.
Expansion candidates that cannot fit current load are recorded as skipped;
they do not hold the entire batch waiting for a larger allocation. The minimum
baseline wait also consumes the timeout and can be cancelled. Recommendations are the
best constrained measured settings, not proof of a global hardware optimum.

In adaptive mode, background pressure stops new admissions first. Running trials
pause/restart only after a complete update, validation and native commit. The
runner restores learner/optimizer/RNG/replay/target-clock/best/patience/collector
state, rather than starting from weights alone. Growth uses measured profiles,
a cooldown and a restart-cost test. Long updates cannot instantly release CPUs.
Fixed mode adopts the calibration result and preserves per-trial threads;
fixed resumes reuse that calibration, while adaptive resumes recalibrate only
eligible unfinished experiments. Resource admission still checks current capacity. No other applications or
research processes are automatically terminated.

The amber performance-evaluation display shows elapsed and remaining wall time,
the current probe phase and candidate, tested/untested counts, resource wait,
queued/running/finished experiments, physical
training ticks, current validation/evaluation cases, requested versus acknowledged
CPUs and allocation epochs. `--log-format json --progress off` writes the command's
JSON result to stdout; stderr contains progress events and Ray diagnostics.
`smartsom monitor <batch-dir>` reads
the saved progress snapshot. An interactive `tune run/recommend` starts a
verified background driver and attaches the Rich view by default. `d` leaves the
task running; `p` reduces optional full-smoke coverage; two Ctrl+C presses
within three seconds request cooperative stop. Explicit `--background` returns
without attaching. The preflight smoke checks executability, while calibration
alone measures throughput and selects the formal resource layout.

A batch directory contains `plan.json` (frozen inputs/source), `batch.json`
(driver-owned ledger), `calibration.json`, isolated `calibration/probes/`,
and retained `calibration/reports/` for every calibration attempt,
`inputs/`, `experiments/<id>/attempt-*/`, paired `controls/`, and separate Ray
segments under `ray/`. `commit.json` and its checksums establish completed update
or experiment boundaries; mutable progress files alone do not. Source/dependency
or scientific-input drift rejects resume. Legacy strict resume is unchanged.

Primary contracts: [Ray Class Trainable](https://docs.ray.io/en/latest/tune/api/trainable.html),
[ResourceChangingScheduler](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ResourceChangingScheduler.html),
and [Ray resource semantics](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html).

See [engineering validation and its limits](validation/ray-tune-execution.md)
before treating these checks as hardware or research-performance acceptance.


Shared queue exit also covers Ray's startup window. The Ray 2.58 adapter journals
the handle immediately after `Actor.remote()` and before the ready future, since
startup cancellation may suppress all start/stop/error callbacks. A handed-off
reservation is retained until the actor is authoritatively DEAD and owned children
are verified dead. Successful cohort cleanup can release a request only when its
complete dispatch journal proves that this request never created an actor. Unknown
identities fail closed and retain the shared slot; a callback's absence is not exit
evidence.

Newborn process CPU counters use the monotonic interval between aggregate samples.
Their wall-time birth must also fall in a consistent wall-clock window. Backward or
forward clock adjustments leave new rows unavailable until a normal second sample,
so clock changes cannot inflate usable capacity by over-deducting owned CPU.
