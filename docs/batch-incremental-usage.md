# Select and resume a frozen batch

The Tune migration described in [ADR0087](decisions/0087-official-tune-execution.md)
is under engineering acceptance. Existing runs remain bound to their original
source, dependencies and commands; no running experiment is migrated in place.

A directory can contain named Experiment files alongside Scenario and Algorithm
files. Only Experiments are tasks. Keep full scientific seed lists in the authored
files, for example `train_seeds: [4101, 4102, 4103]`, and select execution separately:

```sh
smartsom run experiments/my-batch --profile server \
  --id medium-sl --select 'train_seeds: [4101]' \
  --id medium-base --select 'train_seeds: [4102, 4103]' \
  --output-root runs/my-batch
```

Each `--select` binds once to the preceding `--id`. Only `train_seeds` is accepted.
Lists specify actual seeds; an integer retains YAML count semantics (`3` expands
to `[101, 102, 103]`). Use `[4101]` for seed 4101. Each seed must belong to the
frozen Experiment. `--first-train-seeds 1` selects its first authored seed instead.
Local selectors conflict with first-N and scientific seed overrides. Repeated IDs
may select disjoint subsets; duplicate tasks or unknown IDs/seeds fail before
allocation. Selection order identifies the manifest; Tune determines start and
completion order. There is no task priority or `--order` control.

The initial selection freezes the complete inventory. Once its driver exits,
start previously unstarted work from that same inventory:

```sh
smartsom run experiments/my-batch --profile server \
  --resume-missing --output-root runs/my-batch
```

This selects the whole frozen directory, rather than only the earlier subset.
Completed results are skipped. `--id`/`--select` can restrict the remaining work.
A completed selection with unstarted tasks reports `partial`; preallocated task
directories are not trained models. Results use `trials/ENTRY/seed-N` paths.

`--resume-missing` does not recover started trials. Use whole-run `resume` for a
paused/interrupted execution, or `retry RUN --all` to include failed fixed trials.
A training-complete trial with unfinished final evaluation also requires restore.
Missing/corrupt recovery state fails explicitly. Search runs retain terminal
failures and do not support in-place `retry`; unfinished search work uses `resume`.

Scientific `batch.stage` remains a dependency barrier. Tune controls admission
and rolling replacement within each ready stage. Configure the optional trial
concurrency ceiling and evaluation worker target before launch. Logical total
and phase resource requests belong in the profile.

Active owners reject a competing submission. Adding/removing members or seeds,
changing scientific configuration, source/dependencies or directory binding
requires a new run. Online `add/close/configure/remove`, task-level controls and
priority operations are removed. Historical inventory extensions remain readable;
new Tune runs do not write such extensions or mutate frozen manifests.
