# Native reservation Base experiments

The reservation Base is a selectable project contract using the existing grid
simulator. Its Factory/Workload/Algorithm/Experiment inputs live in
`configs/base-diagnostic/`. Historical filenames remain usable; no external
patch script or custom simulator is needed. Both scales use pre/post capacity 4,
8/20 machines and AGVs, and 128/320 orders respectively.

```sh
uv run smartsom check configs/base-diagnostic/small.yaml
uv run smartsom run configs/base-diagnostic/small.yaml \
  --set experiment.run.smoke_ticks=1000 \
  --set experiment.run.output=/tmp/base-small-smoke
```

The default task is a bounded smoke: rule, random and fresh PPO share the scenario
seed and fixed Mover; PPO performs an update and saves `last.pt`. This checks
execution, not convergence or a trained-policy comparison. Use a new output
location for each run. Target scale uses `target.yaml`.

Run settings belong in Experiment YAML, for example:

```yaml
run:
  task: smoke
  output: ../../runs/base-small-smoke
  seed: 101
  seconds: 120
  smoke_ticks: 1000
  methods: [rule, random, ppo]
```

Paths are relative to the Experiment file; absolute paths are accepted. Four
namespaces are supported by `--set`: `factory`, `workload`, `algorithm`,
`experiment`. Unknown fields fail validation. For rules without Torch, use
`--set 'experiment.run.methods=[rule,random]'`. Check never imports Torch.

The same API is available in Python:

```python
from smartsom import api

api.check_experiment("configs/base-diagnostic/small.yaml")
api.run_experiment(
    "configs/base-diagnostic/small.yaml",
    sets=[
        "experiment.run.output=/tmp/base-small-smoke",
        "experiment.run.smoke_ticks=1000",
    ],
)
```

## Calibration, training and evaluation

Set `run.task: calibrate`, `run.seconds`, `run.calibration_cases` and
`run.max_drain` for bounded calibration. The initial candidate remains
`drain_ticks: 2000`; no example silently freezes or extends this limit. A completed
calibration writes `calibration.json`. Then set `run.task: train`,
`run.calibration: /path/to/calibration.json`, `run.seed`, `run.seconds`, and
`train_ticks`. Each native run executes one training seed; run separate outputs
for the three prescribed seeds.

Evaluation uses `run.task: evaluate`, the same calibration, `run.checkpoint`,
`run.methods: [rule, random, ppo]`, `run.selection: sample` or `argmax`,
`run.split: validation` or `test`, and `run.action_seeds`. Repeat with the other
selection in a separate output. World seeds are paired across methods and action
selection. Default validation/test sizes are 32/64 balanced across four conditions.
Final metrics are withheld when a condition has unfinished cases; observed
partial shipment quality is separately labeled `observed_passing_rate`.

Changing execution paths, seed, budget or task does not invalidate native
calibration. Changing factory, workload, algorithm parameters or the initial
drain setting does. Legacy diagnostic commands retain their stricter byte-hash
calibration contract. The old module launch command remains a thin compatibility
entrypoint over the same runtime.

## Evidence and current boundaries

Outputs contain frozen `inputs/`, `manifest.json`, `authoring.json`, progress,
case/episode records, summary or checkpoint as appropriate. Manifest source
identity includes commit, dirty state and Python source digest. Development
smoke on a dirty branch is not a formal main/tag experiment. No uncompleted
smoke is reported as a complete mean tardiness result.

Foreground/background runs and cooperative `smartsom stop DIRECTORY` use common
project control. Training/case continuation and adaptive performance Tune are described
below. General v4 matrix workflows, live elastic scheduling and Studio policy
editing remain separate integrations. Original experiment/model trees, the PPO
learning objective and legal STOP remain unchanged.

The integrated main includes PR27. On this protocol, shared core optimizations
remain available; actor-only evaluation and update-local raw tensor reuse are
explicit Base hot-path adaptations. Raw tensor reuse does not cache learned
embeddings or state across decisions. See [ADR0054](decisions/0054-native-reservation-execution.md).

## Background, continuation and performance Tune

ADR0055 supersedes the earlier foreground-only limitation:

```sh
smartsom run configs/base-diagnostic/small.yaml --task train --background \
  --set experiment.run.calibration=/absolute/calibration/calibration.json \
  --set experiment.run.output=/absolute/new-run
smartsom resume /absolute/new-run --background
```

Training resumes at complete rollout/update transactions. Models, optimizer,
RNG, counters and episode records commit together. A partial rollout is replayed;
wall-clock downtime/replay consumes the original deadline. `run.max_failures` defaults
to one automatic exception recovery; stop/invalid contracts never trigger retries.
Process death of a standalone background driver needs `resume`. Existing input/source
identity must match. Old diagnostic runs lacking continuation generations cannot be
upgraded in place. Completed runs return their existing result.

For optional Ray 2.58 performance calibration use `--tune recommend` or `--tune auto`
with a train configuration. `run.calibration_seconds` (default60) bounds resource
probes, separate from the `run.seconds` training deadline. `run.probe_ticks` (200)
controls probe work and `run.max_concurrent` (2) caps independent seed concurrency.
`auto` runs `Experiment.train_seeds`; each receives `train_ticks`, while the batch shares
one wall deadline after calibration, including queueing and downtime. Queueing can
therefore leave a seed with fewer committed ticks; reports must retain that evidence.

Set `run.resource_execution: fixed` to keep the calibrated per-worker allocation
throughout Tune execution. The default `adaptive` permits measured resizing. Both
use the existing broker admission and the same ReservationSession checkpoints;
fixed mode retains cooperative controls but does not pause merely for changing
host pressure after admission. `run.tune: off` uses the direct native session.

Measured CPU profiles vary threads/concurrency only. ADR0056 enables live elastic
scheduling among those profiles; GPU execution remains unsupported. This does not
change PPO hyperparameters. The existing probe,
resource-observation and background mechanisms are reused. Native generations remain
on the local filesystem; Ray recovery references them and is not portable restore.
Inspect `calibration-performance.json`, `tuning.json`, `summary.json`, and
`seeds/<seed>/progress.json`. Formal experiments still require integrated main/tag;
this dirty development worktree is for bounded verification only.

Native checkpoints commit every rollout/update. Ray reporting checkpoints are saved
every ten iterations and at trial end; worker setup restores the latest native
generation even if it is newer than Ray's reporting checkpoint. This avoids an
extra experiment-state snapshot after every short rollout.

## Elastic execution and stage recovery (ADR0056)

The fixed-profile limitations above describe ADR0055. ADR0056 extends `--tune auto`
to the existing AdaptiveBroker and SafeResourceChangingScheduler: measured CPU
alternatives may resize at complete rollout/update boundaries, and live pressure
can pause trials then readmit them when headroom returns. `max_concurrent` is still
an upper bound. No change to environment count, scientific settings or seed budgets.
Resumption creates a new execution segment from native state, following the project's
preset scheduler. See `elasticity.json` and `seeds/<seed>/allocations/` for requested
versus acknowledged allocations, PID birth and restored counters. A smaller/larger
allocation is not claimed solely because a request was issued.

`smartsom resume DIRECTORY` now also recovers unfinished evaluation and drain
calibration. Complete cases have checksummed receipts under `case-commits/`;
completed and truncated scientific outcomes are skipped equally. An interrupted
case replays from its frozen seed. Model/input changes or corrupted receipts reject
recovery. The original wall deadline includes downtime and replays. This is not
mid-simulation recovery and does not expand a calibrated drain limit.

Performance probes have durable receipts under `probes/receipts/`. Resumed resource
calibration labels reused measurements and keeps their original timing evidence;
missing probes use only the original remaining deadline. `--tune recommend` still
only measures and recommends. An exhausted deadline cannot be reset by resume.

Engineering design and primary-source comparison: [ADR0056](decisions/0056-base-elastic-and-case-recovery.md).

## Configurable platform (development integration)

See [ADR0057](decisions/0057-configurable-reservation-platform.md). The four files
remain Factory, Workload, Algorithm and Experiment. Algorithm can add:

```yaml
observation_schema: native-v2
allow_stop: true
allow_empty_wait: false
learning_reward_scale: 1.0
critic_initial_scale: 1.0
method: ppo
bindings:
  pre.accept: rule
  post.service: random
  dispatcher.pickup: custom
providers:
  custom:
    path: custom_policy.py
    factory: create_policy
```

These are additions to an Algorithm carrying `schema: smartsom.base-algorithm/v1`.
Provider paths are relative to Algorithm. The compiler computes and freezes the
source hash; an authored `sha256` additionally requires that exact source. A
standalone provider file implements:

```python
class Policy:
    def choose(self, request):
        return request.candidates[0].identity

    def get_state(self):
        return {}

    def set_state(self, state):
        pass


def create_policy(*, seed):
    return Policy()
```

Provider state must be JSON serializable. Treat a provider as trusted Python code.
Only its standalone source is copied; dependencies must already be installed and
are not a mechanism for reading private simulator state. Numeric field descriptors
are in `smartsom.domain.base_observation`. Existing builtin rules/PPO retain their
semantic tuple adapter internally. Custom providers return candidate IDs, never
candidate positions. `run.methods: [configured]` selects the authored bindings for
smoke/evaluation. `[rule, random, ppo]` deliberately selects unmixed baselines;
training always uses the authored bindings.

Experiment can explicitly select:

```yaml
train_episodes: 2
condition_sampling: balanced_cycle
budget_boundary: cut
checkpoint_every_episodes: 2
validation_cases: 4
validation:
  milestones: [1, 2]
  modes: [sample, argmax]
  sample_action_seeds: 1
  selection: sample_complete_cost
  select_initial: false
run:
  task: train
  output: ../../runs/base-example
  calibration: ../../runs/base-calibration/calibration.json
  seconds: 120
  diagnostics: false
```

An episode budget supersedes `train_ticks`; omit it to use physical ticks.
`budget_boundary: complete_episode` only affects tick budgets and permits an
overshoot. Recovery commits every rollout/update. Periodic exports appear under
`checkpoints/`; continuation generations remain retained. Validation writes
recoverable per-mode panels and `selection.json`; `best.pt` exists only after an
eligible candidate is selected. Metrics with unfinished demands stay censored.
This example is an engineering check, not the 1000-episode Mac/server experiment.

For a batch, place only Experiment YAML files in a directory, with references to
shared Factory/Workload/Algorithm files outside that directory. Give each
Experiment a distinct seed or scientific configuration and a `run.output` with
the same parent directory. Add `batch: {stage: 0, parallel_files: 2}` to run
independent cases concurrently through the existing scheduler:

```sh
uv run --no-sync smartsom check path/to/experiments --progress off
uv run --no-sync smartsom batch-run path/to/experiments --max-concurrent 2 --no-background
uv run --no-sync smartsom resume /path/printed/by/batch-run
```

Batch children use `run.tune: off`; select shared adaptive Ray execution through
standalone `run` when needed. Existing V4 directory behavior remains separate.
Completed children remain completed on resume; failed children require explicit
`--retry-failed`. An expired original deadline cannot be extended by resuming.

Change configuration for existing permissions, budgets, scales, bindings and
validation. Write one provider file for a new public-observation scheduling rule.
Changing physical capacity, service time, movement, legal candidates, information
visibility or reward accounting requires a reviewed core/contract change. Current
native observation versions explicitly do not claim checkpoint or byte-level
compatibility with frozen server C or the Mac reconstructed-input variant.
