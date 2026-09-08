# Paired study and recovery acceptance

The first study interface supports cases × algorithms × replications × explicitly
listed module-disable variants. No arbitrary parameter grids, distributed jobs,
learning frameworks or in-engine checkpointing are implemented.

```sh
uv run smartsom plan configs/studies/quality_compare.yaml
uv run smartsom batch configs/studies/quality_compare.yaml --workers 2
uv run smartsom batch configs/studies/quality_ablation.yaml --workers 1
uv run smartsom batch --resume PATH_TO_STUDY
uv run smartsom batch --resume PATH_TO_STUDY --retry-failed
uv run smartsom run PATH_TO_RUN/resolved_run.yaml
```

No run directory is created by `plan`. Resolved snapshots embed all domain inputs,
so moving/deleting original authoring files does not prevent recovery or standalone
reruns. A standalone rerun creates its own unique run directory and is not counted
as another batch replication. `run.json` binds each batch attempt to its actual
worker-produced run. Failed runs remain visible and CLI exits nonzero.

Study directories contain `plan.json`, `snapshots/`, `manifest.json`, persistent
lock files, `children/<entry_id>/attempt-*/`, `progress.log`, and JSON/CSV/Markdown
summaries. Failed metrics remain missing. A finished study with failures is not
an integration acceptance pass. Recording options are `recording.observations:
full|hash` and `recording.debug: false|true` in run/study YAML. The study default is
hash, while omission in standalone runs retains their prior behavior. Debug is
rotated at 10 MiB with two backups. Progress distinguishes processing, output and
public inspection counts. CP progress reports stage/elapsed time, not a percent.

`tests/unit/test_studies.py` checks materialize-once pairing, module disabling,
seed golden values, reorder/cwd/hash-seed stability, strict embedded snapshot
validation, preservation of old single-run results, full/hash context equality,
serial/two-process evidence equality, explicit failure retry, partial restart,
source and evidence mismatch refusal, coordinator/worker locks, and actual
subprocess Ctrl+C drain/interruption/recovery. Writer errors retain failed evidence.
The seed golden comes independently from the literal SHA-256 payloads in ADR 0009.

Validation commands:

```sh
uv run pytest -q tests/unit/test_studies.py tests/unit/test_experiment_boundaries.py
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run --extra cp pytest -q
uv lock --check
uv pip check
```

No throughput/large-scale claim follows from these small cases. Full trace memory
and observation-hash computation still cost CPU/memory. Ubuntu runs these same
tests in existing base/CP CI; without an actual CI run Linux remains unverified.

Fresh precommit macOS gate: 20 study tests; base 758 passed / 12 CP skips;
locked CP 770 passed / zero skips. Ruff, format, lock, both dependency environments
and diff checks passed. Local logs: `artifacts/idetc/study-checks/`.
