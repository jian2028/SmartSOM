> Historical workflow record. Current commands and supported configuration are documented in [current-workflow.md](current-workflow.md) and ADR0058; reproduce older examples using their archived source.

# Configurable experiment execution

Use the existing Factory, Workload, Algorithm and Experiment author files. The
options below extend their contracts; see [ADR 0036](decisions/0036-configurable-experiment-contract.md).

In each learning role's Algorithm policy, select the physical-job implementation
explicitly. Retain the existing `network` actor/critic branches for PPO or `q`
branch for DQN to choose hidden widths and ordinary branch settings:

```yaml
kind: new_model
extensions:
  observation:
    name: builtin.physical_job
    version: "1"
    parameters: {include_inspection: true}
  network_implementation:
    name: builtin.physical_job_candidate
    version: "1"
  network:
    actor: {hidden_sizes: [128, 128]}
    critic: {hidden_sizes: [128, 128]}
```

The encoder supports the existing eight/sixteen-machine layouts with finite input
capacities 10/20 and finite physical pre/post storage. It estimates reference work
from released routes and public progress. Eight aggregate summary fields are
critic-only; actor and DQN Q scores mask them. Inspection true/false instances
can coexist. Configuration preflight pins both selectors to implementation source
hashes and rejects incompatible pairs before allocating a run. Packages retain
the declared selectors and physical-job schema. Existing per-role model selectors
can initialize from compatible packages using weights only; optimizers, replay
and named policy RNG are fresh. Full resume is a separate operation.

An Experiment can declare nonuniform dev milestones and an unselected outcome:

```yaml
validation:
  every_updates: null
  updates: [8, 32, 64]
  best_mode: all_complete
evaluation:
  checkpoint: best
  no_eligible_best:
    outcome: record_unselected
    last_diagnostic:
      enabled: true
      label: FINAL-LAST-DIAGNOSTIC
checkpointing:
  retention: {mode: latest_full_and_best}
execution:
  calibration_level: off
  scheduling: fixed
  max_concurrent: 2
```

With 256 physical ticks/update these dev milestones occur at 2048, 8192 and
16384 ticks. The training budget must reach every declared milestone. Keep
`save_best` enabled. If no dev model qualifies, the scientific selection record
has `status: no_eligible_best`, `selected_checkpoint: null`, `final: null` and
`ranking_eligible: false`. Omit `last_diagnostic` to skip final rollouts entirely.
If enabled, the latest FULL model is evaluated only as the labelled diagnostic;
its results and summary are separate from primary final evidence. Diagnostics
still fail on engineering errors. A missing selected model remains an error.

Bounded retention keeps the latest complete FULL continuation, selected best
inference model and any explicitly requested initial/control inference dependency.
An atomic manifest determines which update is committed; aliases are secondary.
Garbage collection affects only generations owned by this opt-in run. Legacy
history is preserved. Tune and Ray keep bounded copies of the latest boundary;
their storage tiers may duplicate that boundary. Readers hold a lease while
using retained artifacts. Model selectors copy retained inference inputs under that
lease into a private local input directory, so later collection cannot invalidate
a prepared consumer. These directories contain inference dependencies only.
Identical payloads share one immutable copy owned by the preparing process until
its normal exit, after live multiprocessing children join; failed copies are
removed immediately. Spawn workers also release their pools on normal return.
Prepared consumers remain
valid for that process lifetime. Execution archives dependencies into its run
before starting workers, so saved runs and their readers do not depend on a
temporary preparation pool. Published retained generations and their GC ownership
remain separate from these process-owned preparation inputs.
Evaluation releases the source lease after archiving its inputs, before playback.

Explicit validation persists pending and completed milestones in FULL recovery
state. Resume completes pending validation before collecting another update,
including at the final training boundary. Reported rounds count completed work.
Cooperative stop checks explicit validation at episode/tick boundaries. It
publishes the completed training update with its unfinished validation milestone
and lets the Tune actor exit normally. A fresh process resumes that milestone
without repeating sampling or optimization; partial dev rollouts never count as
completed validation or replace the selected best model.

Publication synchronizes files and nested directories before replacing recovery
pointers. Collection checks the produced-file inventory and preserves generations
with added or changed content, recording the reason locally. POSIX directory
synchronization is attempted; Windows reports unsupported namespace synchronization.
Process-fault tests exercise publication boundaries, without certifying power-loss
recovery on a particular filesystem.

Run a directory of Experiment YAMLs through `smartsom batch-run DIR
--calibration-level off --no-background`. Put supporting Factory/Workload/Algorithm
YAMLs outside the scanned directory. Fixed scheduling preserves frozen thread
allocations; live resource guards and conservative startup still apply. The
declared concurrency is a ceiling. No calibration measurements are made when off;
input checks, engineering smoke, learning and source-freeze guards remain active.

New options select the versioned semantic identity contract automatically. To
compare retention policies under that same contract, explicitly set
`interface_contract: smartsom.configurable-experiment/v1` on both Experiments.
Raw author documents and hashes remain in provenance. Default configurations
retain their old serialization and identities.

For exact externally prepared worlds, select the opt-in frozen input contract:

```yaml
factory: factory.yaml
seed: 101
training_environment_seeds: legacy_policy
frozen_inputs:
  schema: smartsom.frozen-experiment-inputs/v1
  training:
    world: training-world.json
    data_seed: 707
    environment_seed: 101
  validation:
    - id: development_001
      world: development-world.json
      data_seed: 808
      environment_seed: 303001
      policy_seed: 503001
  evaluation:
    - id: held_out_001
      world: evaluation-world.json
      data_seed: 909
      environment_seed: 404001
      policy_seed: 604001
```

Each JSON or YAML world document has `schema:
smartsom.frozen-production-world/v1`, a complete native `scenario` snapshot and
optional `provenance` mapping. Snapshot seeds must equal the declared actual
environment seeds, and every factory must match the selected Factory file.
Cases run in declared order with replication zero and unique IDs per phase.
`data_seed` records dataset provenance; it is not hashed into another environment
seed. `policy_seed` reseeds native action streams after weight construction.
Frozen case results and live monitor updates expose `data_seed`, actual
`environment_seed` and `policy_root` separately. Their retained `seed` field is
explicitly marked `seed_semantics: data_seed`. Historical case/result fields
remain unchanged. Default frozen-case captions label all three streams.

Omit `workload`, `matrix`, root `data_seed`/`world_seed`/`environment_seed`,
`runtime.environment` and phase `replications` when using frozen inputs.
Validation cases must match enabled training validation. Training requires an
explicit episode rule: `environment` uses the frozen actual environment seed as
root, `policy` uses the training policy root, and both use the existing
`environment:{index}:{episode}` domain. `legacy_policy` explicitly selects the
V3 `training:{index}:{episode}` domain and seed-streams/v2. Ordinary generated
worlds can also select that rule; other existing selections remain v1.

Complete worlds, input file hashes, case IDs, roots, provenance and episode rules
enter the semantic identity and archived run/model contracts. Full resume rejects
an identity change before mutable state is loaded. Weight initialization across
compatible held-out worlds remains available. See [ADR0043](decisions/0043-frozen-worlds-and-direct-case-streams.md).

An Algorithm's learner may explicitly select a training-only passing potential:

```yaml
learner:
  backend: rllib
  algorithm: dqn
  gamma: 1.0
  reward:
    learner_scale: 1.0
    passing_rate_smoothing:
      schema: smartsom.passing-rate-training/v1
      pseudocount: 9
      terminal_compensation: objective_episode
```

Use this with the Experiment's explicit shipment `task_reward`. Only training
passing feedback changes: weight×delta G/(S+9), plus weight×(G/S-G/(S+9)) once
at the genuine objective episode end. Evaluation, initial/rule controls and best
selection retain the original J. An update/drain, dev or checkpoint does not pay
the compensation; a budget cutting an unfinished episode leaves it pending.
The task time limit is an objective terminal even if its status is truncated.
Private action/episode logs and `reports/training-reward.json` distinguish the
smoothed component, terminal compensation, training return and original target.
The unique physical ledger and contract are part of full continuation. See
[ADR0044](decisions/0044-training-passing-potential-with-terminal-closure.md).

### Timing over frozen worlds

To change timing while preserving directly seeded worlds, add an optional native
`frozen_inputs.timing` declaration. It uses the same validated distributions as
[Workload timing](four-file-workflow.md#workload-timing-distributions):

```yaml
frozen_inputs:
  schema: smartsom.frozen-experiment-inputs/v1
  timing:
    arrivals:
      initial_jobs: {capacity_ratio: 0.8}
      interval_seconds: {distribution: exponential, mean: 30}
    due:
      nominal_work_factor: {distribution: uniform, low: 1, high: 3}
  training:
    world: training-world.json
    data_seed: 707
    environment_seed: 101
  evaluation:
    - id: held_out_001
      world: evaluation-world.json
      data_seed: 909
      environment_seed: 404001
      policy_seed: 604001
```

Each training/validation/evaluation reference can also declare `timing`; that
complete object replaces the shared default, rather than merging partial fields.
Use a reference-specific `timing.arrivals.initial_job_ids` with the complete
arrival/due declaration when Job IDs differ by world or an exact initial cohort
mix is required. The default prefix selection does not infer common/novel labels.
A timing declaration must enable continuous arrivals or a nominal-work due factor;
an empty declaration fails rather than silently ignoring legacy scalar fields.

The public compiler reads the existing native snapshot and applies timing to its
existing demands. It permits only release/reveal/due changes and an explicitly
selected initial ordering. Routes, IDs, machine duration overrides, priorities,
rush labels, input IDs, factory, processing/quality samples, outages, transport,
mode, horizon, reward settings and all declared roots remain intact. Each world's
direct `data_seed` derives independent timing RNG domains. No Job pool, physics
sample, environment seed or policy root is regenerated. The frozen training
`legacy_policy` episode rule remains unchanged.

Preparation freezes the exact resulting table and timing evidence in ordinary
workload provenance and in the frozen-world continuation recipe. The input
contract retains `world_sha256` as the original source-file hash and adds
`materialized_world_sha256` for the actual resulting native world. Timing evidence
also records the original-demand hash. Source provenance, including any original
permutation/V labels, explicitly describes the original world before timing;
it does not claim a recomputed V for a newly selected initial order. Archived
continuation consumes the materialized frozen world and never reapplies the
sampler. Omitting timing retains historical serialized inputs and identities.
