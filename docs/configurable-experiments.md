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
using retained artifacts. Windows reports its namespace durability limitation.

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
