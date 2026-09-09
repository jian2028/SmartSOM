# Batch learning and declared searches

These APIs orchestrate the existing training lifecycle. They do not change the
simulator or the historical `StudySpec` / `run_batch` simulation-study contract.
Every child has its own real run directory, model files and evidence.

## A batch of explicit recipes

```python
from smartsom.api import batch_train, load_preset

first = load_preset("sb3_micro")
second = load_preset("marl_micro")
result = batch_train([first, second], max_concurrent=2)
print(result.run_dir, result.completed, result.failed)

restored = batch_train(resume=result.run_dir)
```

A batch does not require validation or select a winner. `max_concurrent` bounds
independent training processes. Each child's environment count, sampling workers
and numerical threads retain their own explicit settings. The coordinator uses
spawned processes; it does not share models, optimizers or random generators.

The CLI uses the same public API:

```sh
smartsom batch-train --preset marl_micro --seeds 101 102 --max-concurrent 2
smartsom batch-train --recipe experiment-a.yaml --recipe experiment-b.yaml
smartsom batch-train --resume runs/SAVED_BATCH
smartsom search --config search.yaml
smartsom search --resume runs/SAVED_SEARCH --retry-failed
```

`--set search.space=...` accepts the same strict typed search declaration as YAML.
Resuming uses the frozen study and rejects recipe overrides. Exit code 1 denotes
failed or pending work, 130 denotes interruption, and 2 denotes invalid inputs.

## Grid, random and Optuna

Search requires an explicit objective and failure policy. It scores the fixed
validation at the last training update, bound to that checkpoint's weights and
checksum. Training evidence must pass `audit_training`; every validation input
must complete and have a finite value. Failed and incomplete cases have no score.
Final evaluation data is never a search objective.

```python
from smartsom.api import load_preset, search
from smartsom.config.experiment import CategoricalSpace

config = load_preset("sb3_micro")
config.search.method = "grid"
config.search.space = {
    "algorithm.learning_rate": CategoricalSpace(choices=(1e-4, 3e-4)),
    "algorithm.hidden_sizes": CategoricalSpace(choices=((64, 64), (128, 128))),
}
config.search.seeds = (101, 102)
config.search.objective = "makespan"
config.search.direction = "min"
config.search.failure_policy = "all_complete"
config.runtime.max_concurrent = 2
result = search(config)
```

Parameters must be public fields under `algorithm.*` or `training.*`, except
`algorithm.source`, which stays fixed; compare providers with an explicit batch. Validation
and evaluation seeds, output paths and runtime settings cannot be searched. Each
candidate uses the same declared training seeds; each training seed is recorded
separately. The candidate score averages its eligible seed results only after all
seeds pass. Training must save `last`, and the validation interval must divide the
planned update count. The API refuses mismatches instead of changing the budget.
Early stopping records the actual final update and budget coverage.

Grid search accepts categorical choices and expands the complete Cartesian
product. Leave `search.trials=None`, or set it to that exact product size. Random
search additionally accepts `FloatSpace` and `IntSpace`, including explicit log
sampling, and requires a trial budget. Its private RNG uses `search.seed`; all
random candidates are persisted before training and do not consume training RNG.
Duplicate random proposals count as declared trials; they are not silently replaced.

For Optuna, install the optional locked dependency and change the method:

```sh
uv sync --locked --extra learning --extra cpu --extra search
```

```python
config.search.method = "optuna"
config.search.trials = 12
config.search.seed = 42
config.search.pruning = False
result = search(config)
```

Optuna uses real `TPESampler` ask/tell with a local SQLite study. The sampler seed
is derived from the search seed and proposal index. TPE's initial random startup
is retained; a short search is not evidence of adaptive optimization. Structured
categorical values use stable categorical indices. The Optuna version is recorded
and a changed version is rejected on resume.

Setting `pruning=True` enables `MedianPruner`. Intermediate values come from
all-complete validation at matching update numbers across the declared training
seeds. With multiple seeds, pruning can begin while the final seed is running;
results from earlier seeds remain available at each matching update. A prune
request saves the current update and returns status `pruned`. Pruned candidates
have no final objective and cannot become `best_trial`. Pruning is off by default.
Validation replay is recorded according to `validation.full_replay`; a score does
not imply that optional full replay was enabled.

## Evidence and recovery

The outer directory contains `run.json`, an immutable `config/plan.json`,
`trials/<id>/plan.json`, and separate `attempt-000`, `attempt-001`, ... directories.
The frozen trial plan contains the typed recipes and resolved scientific inputs.
`summary.json` and `summary.csv` report completion, failure, pending and pruning
states. A winning trial appears only when an eligible score exists. Per-seed rows
retain their own checkpoint, validation digest, completed input IDs and audit.

```python
restored = search(resume=result.run_dir)
retried = search(resume=result.run_dir, retry_failed=True)
```

Resume verifies the code commit, executable source digest, Python/dependencies,
plan checksums and completed evidence before reuse. It never silently accepts a
modified recipe. A lost worker remains interrupted. If a saved update is usable,
the existing training API restores it. If an interrupted child has no update,
a new execution attempt retains the old evidence and reuses verified completed
seed runs through explicit paths, without copying their files.

Already generated candidates execute from checksummed `resolved-training/v1`
snapshots through `train_prepared`, including after export/import and removal of
their original authoring files. Snapshot/config hashes, seeds and budgets are
validated before execution; output allocation is the only relocation.
Optuna also freezes one materialized template for each declared training seed
before asking for candidates. Future suggestions bind only the declared PPO or
training parameters to those templates, without reopening authoring files. The
scenario, provider source and training seeds remain fixed. A historical search
plan without templates is rejected with an explicit request to create a new search;
it is never silently upgraded from mutable authoring inputs.

Failed trials are retried only when requested. An Optuna retry enqueues the exact
persisted parameters in a new Optuna attempt; it does not replace them with a fresh
suggestion. An interrupted RUNNING Optuna attempt retains its existing identity.
Already persisted proposals are recovered before asking for another candidate.
An invalid adaptive proposal is recorded as a failed candidate and consumes its
trial slot; it is never silently repaired or resampled.

The first Ctrl+C stops starting new trials and lets active workers finish. A second
Ctrl+C interrupts only workers owned by that coordinator. Persistent OS locks
prevent concurrent coordinators or workers from owning the same study/trial.

All proposals and their snapshots are held in memory for grid/random planning;
this is a bounded local workflow, not an arbitrary-scale distributed scheduler.
The real integration tests cover macOS CPU. Linux, h20 and CUDA execution remain
separate acceptance work.
