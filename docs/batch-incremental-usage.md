# Grow a batch without overwriting prior seeds

This behavior requires ADR0065's implementation. An old run frozen under another
source commit is not automatically upgraded: strict source identity still applies.
Keep its results and arrange a separately reviewed deployment/migration first.

A directory may contain `medium-base.yaml`, `medium-sl.yaml`, `small-base.yaml`
and `small-sl.yaml` (Experiment schema), alongside Scenario/Algorithm files.
The latter are not tasks. Each Experiment keeps its full authored seed list.

```bash
smartsom batch-run experiments/my-batch \
  --id medium-base --id medium-sl --first-train-seeds 1 \
  --order medium-sl --order medium-base --output-root runs/my-batch
```

After completion, adding `large-base.yaml` and `large-sl.yaml` does not alter the
old members' identity. Freeze the added members and execute just the selected one:

```bash
smartsom batch-run experiments/my-batch --id large-base \
  --first-train-seeds 1 --resume-missing --output-root runs/my-batch
smartsom batch-run experiments/my-batch --id medium-base --id medium-sl \
  --resume-missing --output-root runs/my-batch
```

The second command fills missing authored seeds and unfinished final evaluation;
completed results are not rerun. Failed tasks retain receipts and require explicit
retry via the existing resume controls. Changing an old scientific configuration,
removing an old member, or pointing another authoring directory at this output
root is rejected. `plan.json` and `manifest.json` remain original evidence; new
membership is recorded in `inventory/` and the atomic control journal. Results
remain under their original `trials/<experiment>/seed-<actual-seed>` paths.

While a queue is accepting work, a single independent stage-0 Experiment can be
appended with `smartsom batch-add runs/my-batch experiments/my-batch/large-base.yaml
--first-train-seeds 1`. A directory submission can use `--id` and repeated
`--order`. The explicit order must list all selected entries once; it is not a
partial priority hint. New work goes after already accepted work. A closed queue
uses `batch-run --resume-missing`; later-stage append remains unsupported.

Order is an admission preference within stages, not a completion guarantee.
Resource limits, manual holds and runtime/UI priority remain authoritative.
Unselected deferred seeds do not implicitly block the next selected stage, but
previously submitted unfinished predecessors and explicit dependencies do.
