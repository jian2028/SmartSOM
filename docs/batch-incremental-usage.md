# Grow a batch without overwriting prior seeds

This behavior requires ADR0065/0066's implementation. An old run frozen under
another source commit is not automatically upgraded: strict source identity still
applies. Keep its results and arrange a reviewed deployment/migration first.

A directory may contain `medium-base.yaml`, `medium-sl.yaml`, `small-base.yaml`
and `small-sl.yaml` (Experiment schema), alongside Scenario/Algorithm files.
The latter are not tasks. Each Experiment keeps its full authored seed list, e.g.
`train_seeds: [4101, 4102, 4103]`.

```bash
smartsom batch-run experiments/my-batch \
  --id medium-sl --select 'train_seeds: [4101]' \
  --id medium-base --select 'train_seeds: [4102, 4103]' \
  --output-root runs/my-batch
```

Each `--select` binds to the preceding `--id`, once only. It is an execution
subset, not a scientific override: full manifests, task identities and authored
YAML stay unchanged. Only `train_seeds` is accepted. Lists mean actual values;
an integer uses the existing YAML count semantics: `train_seeds: 3` means
`[101, 102, 103]`, **not** the first three authored seeds. Actual seed4101 must be
written `[4101]`, not `4101`. Every expanded value must belong to that Experiment.
Use `--first-train-seeds 1` without local selectors for the first authored seed.

`--id` selects entries and establishes eligible launch priority. Within each
request, selector list order is retained; a bare ID uses the entire authored seed
list. An ID can recur with disjoint subsets to interleave experiments. Duplicate
tasks (including bare ID plus an overlapping subset), unknown IDs/seeds, duplicate
YAML keys, malformed/unknown selection fields, or a select without ID are rejected
before allocation. Local selectors conflict with `--first-train-seeds`, `--seed`
and `--set experiment.train_seeds=...` / `experiment.run.seed=...`. First-N also
conflicts with scientific seed overrides. There is no `--order` parameter.

Without `--id`, every entry is selected in relative-path discovery order, with
its authored seed order. To fill the entire directory after a subset completes:

```bash
smartsom batch-run experiments/my-batch --resume-missing --output-root runs/my-batch
```

This selects from the full authored inventory, not the previous subset. Completed
results are skipped. Training that finished but lacks final evaluation continues
only that evaluation. Failed/interrupted training requires a compatible, intact
continuation; missing/corrupt recovery data blocks submission instead of starting
fresh. Active owners reject another coordinator. Manual paused/stopped/removed
holds require explicit controls; this command never silently releases them.

New `large-base.yaml` / `large-sl.yaml` members can be added to the same directory.
The full-directory command freezes them and fills their missing seeds too. To
select just one new seed, combine `--resume-missing` with `--id` / `--select`.
Changing or removing old scientific inputs, changing code/dependencies, or binding
another authoring directory to the existing output root is rejected. Original
`plan.json`, `manifest.json`, archives and results stay intact; additions are
recorded in `inventory/` and the atomic control journal. Results remain under
`trials/<experiment>/seed-<actual-seed>`.

A single independent stage-0 Experiment can be appended to an accepting queue:

```bash
smartsom batch-add runs/my-batch experiments/my-batch/large-base.yaml --first-train-seeds 1
smartsom batch-add runs/my-batch experiments/my-batch \
  --id large-sl --select 'train_seeds: [4101]'
```

Closed queues and later stages use `batch-run --resume-missing`. Appends join the
existing queue tail without preempting workers. Selection across multiple owning
batch directories is restricted to unambiguous whole-entry submissions.

Order is an admission preference, not a completion guarantee. Stage barriers,
resource limits, manual holds and runtime/UI priority remain authoritative.
Unselected deferred seeds do not implicitly block the next selected stage, but
previously submitted unfinished predecessors and explicit dependencies do.
