# Independent checkpoint evaluation

Evaluation restores an existing inference checkpoint and runs the selected model
on newly seeded repetitions. It does not train or tune a recipe. Source model
payloads remain unchanged; new update checkpoints may receive retention references
outside their signed payloads. The default is seed `202`, five repetitions,
deterministic inference, and full evidence replay. No baseline is added implicitly.

The implementation entry point accepts the public evaluation options object or an
object with the same attributes:

```python
from types import SimpleNamespace
from smartsom.experiments.evaluation import evaluate_checkpoint

result = evaluate_checkpoint(
    "runs/my-training-run",
    SimpleNamespace(
        seed=202,
        replications=5,
        deterministic=True,
        full_replay=True,
        checkpoint="last",
        baselines=(),
        scenarios=(),
    ),
    output_root="runs",
)
print(result.status, result.completed, result.failed, result.run_dir)
```

Accepted sources include a legacy training directory, a legacy `checkpoint`
directory, a controlled `checkpoints/update-*` directory with `inference/`, or an
outer experiment directory whose `run.json` declares `paths.training`. The
`last.json` and `best.json` selection records are respected; a missing selection
is retained as a failed checkpoint-selection stage without launching an episode. An explicit checkpoint directory selects that
directory. A previous evaluation directory can select its recorded checkpoint.
Imported packages use their relocation map, even if the original source directory
still exists.

A model ZIP or full experiment ZIP can be passed directly to the same API or
`smartsom evaluate MODEL.zip`. Its inventory and checksums are verified before
extraction into this evaluation's `evidence/imported-model/` directory. The selected
checkpoint and training snapshot point into those durable files, not a temporary
directory. A model bundle provides one explicit saved model; an experiment bundle
retains its available last/best selection. Backend, structure and extension
configuration come from the checkpoint metadata.

The default case is named `training` and is reconstructed from the retained
`resolved_training.json`. The fixed factory and workload come from that snapshot;
generated arrivals, processing times, machine events, and quality draws use the
evaluation scientific seeds and retain their fresh provenance. Historical
scenario, factory, workload, or profile paths are not reopened. A checkpoint
without a training snapshot requires explicit `scenarios`, each naming a scenario
file or a project directory with `scenario.yaml`. Explicit cases use the file
stem and its content digest as their recorded identity. Checkpoint structural
compatibility is validated before any evaluation run begins.

Set `baselines=("spt", "first_feasible")` for an explicit dispatch comparison.
Additional checkpoints can be supplied as `"checkpoint:/path/to/checkpoint"`.
Each case and repetition is materialized once, then shared by all selected
algorithms. `study_roots(seed, case_id, replication, algorithm_id)` gives each
algorithm its own algorithm and solver streams while preserving the paired world.
Stochastic inference uses the algorithm stream and is recorded as such.

The `cp` baseline is available only for the existing eligible static scenario
contract and installed optional CP dependencies. It receives `full_static`
information while the model keeps its decision context. This is a comparison on
the same physical input with extra information for CP; it is not an observation
matched fairness claim. Dynamic or logistics scenarios are rejected in preflight.
The evaluator does not silently disable modules to make CP eligible.

Each evaluation creates a new directory containing:

- `run.json`: source identity, options, checkpoint identities, case mapping,
  per-run outcomes, and current/final status.
- `plan.json`: the complete planned case, replication, algorithm, scientific seed,
  and input digest matrix.
- `summary.json`: completed/failed counts and per-case, per-algorithm makespan
  statistics, calculated only from completed and verified runs.
- `evidence/imported-model/`: verified model/experiment bundle files when the
  input is a ZIP; the original archive checksum is recorded in `model_input`.
- `evidence/runs/`: the ordinary `run_one` directories with their actual resolved
  input snapshots, manifests, traces, observations, schedules, and failures.

`completed` means every requested run completed and all requested audits passed.
`completed_with_failures` retains legitimate noncompletion, such as policy stall,
deadlock, episode budget exhaustion, or a solver without an incumbent. These
outcomes have no makespan. `failed` means an engineering or evidence audit failure
occurred. Interruptions stop further runs and retain `interrupted` status plus
the completed and pending counts.

Full replay checks artifact digests, semantic action replay, the physical
schedule, operation and output coverage, quality outcomes, and public observation
hashes. Schedule replay may coalesce waits, so its public observations are checked
at corresponding physical decisions, while action replay checks every original
decision observation. Resource MARL additionally checks the joint ledger,
proposal mapping, masks, actions, rewards, and termination. An incomplete run can
only receive `partial_verified` for its retained prefix, with schedule replay
marked inapplicable. A failure before simulation has no execution to replay.
Missing or inconsistent evidence fails the audit; disabled replay is recorded as
`not_requested`.

Passing these checks establishes engineering reproducibility for the recorded
inputs and checkpoint. It does not establish a performance target or promote a
roadmap milestone by itself.

Basic option type/range errors are rejected before directory allocation. Once the
evaluation directory exists, checkpoint reading, ZIP import and scenario/snapshot
preparation failures are saved in `run.json` and `summary.json` with their stage,
error and final `failed` or `interrupted` status. Already imported files and
executed episodes remain available. Exceptions retain their type and carry
`run_dir` pointing to the evaluation root, so a train/evaluate pipeline can record
its failed evaluation stage. A secondary metadata-write failure is attached as an
exception note and never replaces the original error.
