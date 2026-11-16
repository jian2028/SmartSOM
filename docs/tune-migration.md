# Tune migration: implementation and acceptance status

The user-approved core closeout has passed local Linux checks and independent
engineering, scientific-semantics and testing review. This is not a release or
research claim: deferred features below remain unsupported, and macOS/GPU are
not qualified by these checks. Representative PPO/DQN timing evidence covers
its frozen workload, not every possible configuration or public entrypoint.

The subsequent RLlib migration is specified by ADR 0089 and 0090. Its shared
AlgorithmConfig builder, heterogeneous FIFO Tune adapter, official checkpoint
restore, and resident fixed-panel evaluator have engineering tests. They are
now connected to the public PPO/DQN `smartsom run` lifecycle, including ordinary
trials and basic three-objective Optuna searches. Scientific lifecycle and public
CLI tests are engineering evidence, not migration or performance acceptance.
The public scheduler is FIFO with resident components. The old native scheduler,
phase bootstrap and resource-changing callbacks are not compatibility paths.

## Smoke and scientific calibration

Rule-only smoke and scientific drain calibration run in the same Tune controller
as training. These serial tasks reserve the profile's driver resources and create
no Learner, optimizer or replay. Calibration retains Rule/Random cases, their
independent seeds, drain extension, stalled-case diagnostics and strict completion
requirements. Its reports and case receipts are included in Tune checkpoints;
restoring a completed calibration does not rerun its cases.

Learning smoke uses the configured official PPO/DQN Algorithm and the ordinary
RLlib Trainable. Its physical-tick budget is observed at official iteration
boundaries, so the actual work may exceed the requested budget. Rule/Random smoke
cases still use their exact diagnostic tick bound. Learning smoke writes official
checkpoint references, never `last.pt`, and disables periodic/final evaluation.
The learner/observation recipe must already be an RLlib recipe; native learner
parameters and native checkpoints are not translated automatically.

The user-approved [core closeout](decisions/0098-rllib-core-closeout.md) defers
standalone checkpoint evaluation, window/recovery analysis, centralized/Morris
learning adapters and new recordings. ADR0100 enables named Base Rule/Random
fixed panels; formal comparative experiments still require a separate launch.
Public check/run reject these settings before allocating a run. Historical tests
and their replacement/defer mapping remain in `archive/rllib-transition-tests`.

ADR 0091 supersedes the earlier argmax-only best selection and sampled-DQN
constant epsilon override. New panels use the resolved official evaluation
policy: PPO samples, DQN does not explore. Best minimizes mean J over the entire
complete periodic panel under that same policy; no extra argmax panel is added.
No evaluation epsilon override is introduced. Short engineering tests do not establish performance. The separate frozen
three-pair PPO training, PPO evaluation/best-last and DQN training comparisons
met the 5% median steady/end-to-end gates; final quality acceptance is separate.

## Execution configuration

`smartsom check EXPERIMENT --profile PROFILE` validates scientific inputs,
current support restrictions and structural resources without importing Ray. Full official AlgorithmConfig
validation occurs when preparing the optional RLlib runtime. `smartsom run` uses the same
compiler. The local and server profiles retain their authored concurrency caps
of 2 and 4. Without an explicit cap, Tune uses resource availability.

```yaml
resources:
  cpus: 8
  memory_gib: 16
rllib:
  resources:
    num_cpus_for_main_process: 1
    memory_gib: 3  # includes local Learner and local sampling, if used
  learners:
    num_learners: 0
  env_runners:
    num_env_runners: 2
    num_cpus_per_env_runner: 1
    memory_gib: 1  # each training EnvRunner
  evaluation_num_env_runners: 1
  evaluation_config:
    num_cpus_per_env_runner: 1
    memory_gib: 2  # each evaluation EnvRunner
concurrency:
  max_concurrent: 2
  evaluation_processes: 2
  evaluation_cpus: 2
```

Resident requests include the learner/coordinator and child workers. For two sampling
workers and one learner thread, at least three CPUs are required. Parallel
evaluation similarly reserves its coordinator in addition to its case workers.
Requests govern admission; they do not impose OS CPU quotas or RSS limits, and
reserving additional CPUs alone does not increase numerical-library threads.
Memory is optional. When supplied, the implementation uses the official custom
logical resource `smartsom_memory_gib` for the profile budget. It does not impose
RSS isolation or call the hidden `ray.init(_memory=...)` API.
Trials without a memory request cannot be claimed to have memory-limited
concurrency. Total local CPU counts must be whole numbers for the pinned Ray
startup API.

Each trial requests its full resident component allocation. Training and evaluation
run sequentially while their workers remain reserved; there is no bootstrap or
phase-driven actor replacement.

## Removed controls

These fields now produce explicit migration errors:

- `run.tune`, `run.resource_execution`, `run.calibration_seconds`, `run.probe_ticks`.
- `concurrency.mode` and `resources.guard`.
- `batch.parallel_files`: migrate its explicit cap to
  `concurrency.max_concurrent`; conflicting caps require an explicit choice.

`run.calibration` remains scientific drain calibration. It is not a performance
probe. `batch.stage`, fixed evaluation panels, training seeds, scientific budgets,
checkpoint selection and tracking settings retain their scientific roles.

For searches that reference calibration, the original report must match the
unchanged template. All candidates share that validated drain, with the complete
original report, its digest and an explicit derivation record frozen inside each
trial. Changing a learning parameter does not claim a fresh calibration. Original
external report paths are not needed for candidate recovery. Fixed experiments
retain strict calibration matching. See [ADR0088](decisions/0088-shared-search-calibration.md).

`run.max_failures` defaults to zero. `run.seconds: null` remains available for
training/evaluation; restoration retains the original finite deadline when set.

Online `add`, `close`, `remove`, `configure`, `--keep-open` and task-level control
options have been removed from the CLI. Whole-run pause first obtains durable
trial boundaries, then invokes Tune's documented SIGINT shutdown. Resume uses
`Tuner.restore`. `retry RUN --all` enables restoring failed trials. Finished
trials stay finished in the same Tuner experiment.

`--resume-missing` requires the exact frozen scientific inventory, skips completed
work and can select unstarted frozen tasks. It does not add trials to a running
Tuner and does not replace recovery of started/failed trials. Existing historical
run directories are not rewritten by configuration migration.

## Execution and restore provenance

For the RLlib runtime, DQN retains replay in the same official Ray 2.60 checkpoint
through the approved thin state extension
([ADR0095](decisions/0095-same-version-dqn-replay.md)). Same-version recovery
preserves experience and learner state; old-version checkpoints are not migration
inputs. No separate model or replay storage system is maintained, and exact
equivalence to uninterrupted simulator execution is not promised.

Each actor execution has its own ID. Checkpoints carry the producing execution
and its observed history; successful restoration records the checkpoint metadata
digest and the new execution ID. Resource-driven actor replacement is recorded as
a new execution, without classifying it as a failed-trial retry. History survives
scientific rollback and is portable with the checkpoint. Checkpoint contents and
both incoming and existing local history are validated before scientific files
are restored; a rejected restore does not create a successful recovery record.
These observations do not replace Tune's authoritative trial failure status.

## Search interface under verification

The intended entrypoint is `run EXPERIMENT --profile PROFILE --search SEARCH.yaml`.
A search specifies `algorithm: optuna`, `num_samples`, search `seed`, one
`train_seed`, parameter paths/distributions and explicit `objectives`:

```yaml
objectives:
  - {panel: sample, metric: configured.all.makespan, mode: min}
  - {panel: sample, metric: configured.all.mean_tardiness, mode: min}
  - {panel: sample, metric: configured.all.final_passing_rate, mode: max}
```

List order aligns values and directions; it is not priority or weighting.
Results preserve all trials and publish Pareto candidate paths using Optuna's
public Pareto selection. Final test panels are never optimization inputs. Training
reward and best/last checkpoint selection remain unchanged. Optional frameworks
are loaded only on execution paths.

Trials report null objectives until real periodic evidence exists.
Official multi-objective OptunaSearch ignores intermediate reports. Each new panel
replaces the previous objectives, including nulls from a censored panel. Complete
makespan is the last original-order delivery tick, measured from tick zero; a
cutoff is not substituted. An incomplete case makes the corresponding panel
aggregate unavailable rather than calculating a success-only average.

Ray 2.58 forwards null completion vectors to Optuna without terminating its trial.
The approved public `CensoredEvaluationSearch` adapter delegates proposals and
save/restore, but sends censored completion through the official no-result path.
Optuna records PRUNED, Tune records normal completion, and scientific receipts keep
the original truncation reason. There is no application retry or fabricated score.
Budget termination before any periodic panel is explicitly `not_evaluated`, also
without a score. Actual execution exceptions remain failures.

Real correctness probes cover COMPLETE/PRUNED separation and null objectives.
The core closeout test additionally runs a calibrated three-objective PPO search
through public CLI pause and fresh-process resume, preserving completed candidates,
frozen calibration and deduplicated tracking. These are correctness evidence,
not search efficiency or learning-quality claims.

A failed Optuna trial cannot be retried in place: the pinned official searcher
does not turn an already failed ID into a completed observation. `retry --all`
therefore rejects search runs before changing their settings. Preserve the failed
search and start a new one. `resume` still restores unfinished trials and search
state while retaining terminal failures. Fixed experiments retain `retry --all`.

## Evidence and remaining acceptance

The agreed performance scope is one representative configuration for PPO and DQN,
with three alternating direct-RLlib+Tune/public-entry pairs. Both steady and
end-to-end median regressions must remain at most 5%; startup remains in end-to-end
cost. Official replay randomness may produce different trajectories at equal
environment-step and learner-update work. Record that difference rather than
adding RNG adaptations or claiming pure wrapper speedup from differing work.

Retired native lifecycle tests have been mapped to current core scenarios or
explicit historical/deferred coverage. All current Ubuntu owner selectors passed, including a clean dependency
environment for public lifecycle tests. The final Tune owner excludes archived
native wave/update-credit and custom case-pool tests. This local Linux evidence
does not constitute the hosted cross-platform GitHub gate. Existing experiments and historical
artifacts remain untouched; no formal campaign, release or research conclusion
is authorized by these engineering measurements.

Named Base Rule/Random fixed-panel evaluation is enabled by ADR0100 through
`run.task: evaluate` and `run.methods: [rule]` or `[random]` (or both). It reuses
fixed inventories and scientific receipts in the shared Tune Trainable; CPU-only
Ray case workers are reserved with the coordinator. Learning standalone evaluation
and Central remain deferred. Maintained recipes are under `experiments/current`.
