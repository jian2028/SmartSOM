# Ray Tune batch execution

This is an execution tuner for frozen v3 experiments. It uses Ray Tune 2.58.0,
FIFO scheduling and `ResourceChangingScheduler`; it does not search scientific
hyperparameters or terminate experiments because their rewards are low.

Install optional dependencies in the environment you intend to use:

```bash
uv sync --locked --extra learning --extra tuning --extra cpu
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
mode: office
execution: adaptive
active_limit: 600
output_root: runs
entries:
  - id: machine-ppo
    config: configs/test/runs/train_machine_ppo.yaml
  - id: machine-dqn
    config: configs/test/runs/train_machine_dqn.yaml
```

```bash
uv run smartsom tune check --batch batch.yaml
uv run smartsom tune recommend --batch batch.yaml
uv run smartsom tune run --batch batch.yaml
uv run smartsom tune resume runs/<tune-batch> --retry-failed
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

Execution follows: preflight → resource observation → baseline for every workload
group → candidate measurements → repeated leader measurements → recommendation →
queued experiments → complete update checkpoints → final evaluation and controls.
Calibration groups retain algorithm/backend, learner shape, factory, device and
sampling layout and update quantum; each group uses a representative frozen workload. Measurements
include cold initialization, actual sampling/learning, validation and checkpoint
cost, concurrent process-tree RSS peaks and aggregate physical-tick throughput.
Calibration evidence and checkpoints are separated from experiment evidence and
are never used to initialize the formal learner.

Only experiment concurrency and numerical threads are tuned. Requested CPU is
numerical threads plus the original sampling-child thread reservation. Existing
`num_envs`, sampling-process counts/order, models, seeds, workloads, budgets,
scientific parameters and CPU/CUDA cohort remain frozen. There is no newly
implemented parallel sampler or validation-case parallelism. CUDA calibration
currently measures one complete GPU trial at a time; multi-GPU performance and
H20/CARC hardware acceptance remain unverified.

`office` reserves at least one CPU and 20% CPU, and at least 2 GiB and 20% RAM.
`throughput` reserves at least one CPU and 5% CPU, and at least 1 GiB and 10% RAM.
Memory admission uses observed peaks with a 1.25 factor. These are conservative
resource budgets, not operating-system CPU quotas or a measurement of GUI frame
rate. If minimum capacity is unavailable, the display stays in resource wait
until capacity returns or Ctrl-C cancels. **Pure waiting has no automatic timeout**
and does not consume the at-most-600-second active calibration budget. Background
load changing during calibration can therefore make wall-clock time longer than
600 seconds. Unconfirmed leaders at the deadline use a valid measured baseline
with an explicit reason; missing baselines prevent training from starting.
Expansion candidates that cannot fit current load are recorded as skipped;
they do not hold the entire batch waiting for a larger allocation. The minimum
baseline still waits indefinitely and can be cancelled. Recommendations are the
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

The normal display shows independent calibration-active and resource-wait
clocks, candidate progress, queued/running/finished experiments, physical
training ticks, current validation/evaluation cases, requested versus acknowledged
CPUs and allocation epochs. `--log-format json --progress off` writes the command's
JSON result to stdout; stderr contains progress events and Ray diagnostics.
`smartsom monitor <batch-dir>` reads
the saved progress snapshot.

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
