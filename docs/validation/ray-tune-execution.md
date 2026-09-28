# Ray Tune execution engineering validation

Date: 2026-09-27. These are disposable CPU engineering checks on macOS with
Python 3.12, Ray 2.58.0 and the locked learning/tuning/CPU dependencies. The
original production `.venv`, active composable study and its worktree were not
used as test environments. Test evidence was written under `/private/tmp`.
No first daily Small recipe or formal training batch was created.

## Fresh checks

| Check | Completed result | What it establishes |
| --- | --- | --- |
| Full minimal-dependency suite | 1629 passed, 261 skipped; exit 0 | Base compatibility and optional-dependency isolation |
| Final complete tuning unit/integration suite | 232 passed, no skips; exit 0 | All tuning checks together after the portability and optional-runtime fixes |
| Focused tuning/native/v3 suite | 248 passed; exit 0 | Admission, resource observation, calibration, native continuation, real Ray resize and existing v3 learning regressions |
| Final broker/batch regressions | 65 passed; exit 0 | Missing-Ray failure handling, live capacity, completed/failed eligibility and fixed-resume behavior |
| Real PPO fixed batch | Passed | Preflight, real calibration, complete native updates, validation, held-out evaluation, verified commit and completed-resume skip |
| Real DQN adaptive batch and CLI import check | 2 passed; exit 0 | The same batch lifecycle for DQN and read-only JSON preflight of an imported study |

These counts overlap and must not be added together. The real resize tests use a
controlled admission function to exercise `ResourceChangingScheduler` with
native PPO/DQN actors, changing two requested CPUs to one and restoring complete
state. They are not machine-load performance benchmarks. Production broker
capacity/ownership behavior is checked separately with deterministic process and
resource observations. An actual sampling-child test checks lazy child ownership
and release; the existing sampler's scheduling semantics remain unchanged.

Ruff, Ruff formatting, lock consistency and whitespace checks accompany the
change. A bare `import smartsom` in the minimal environment imports none of Ray,
Torch, Gymnasium, PettingZoo or W&B. The CI workflow includes a separate optional
tuning job that requires the real dependencies before running the tuning unit
and integration tests. Its Linux execution is not claimed as completed locally.

## Reproduce the affected acceptance

Use an isolated environment if an existing `.venv` belongs to a running study:

```bash
uv sync --locked --extra learning --extra tuning --extra cpu
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv lock --check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --no-sync pytest -q \
  tests/unit/test_tuning_*.py tests/integration/test_tuning_*.py \
  --basetemp=/tmp/smartsom-tuning-acceptance
```

The acceptance authors tiny frozen engineering recipes **before** preflight.
Calibration itself preserves their budgets, networks, algorithms, seeds and
device. PPO and DQN complete native training/evaluation; probes remain isolated
and never become the batch's initial trained model.

## Limits and retained diagnostics

The earlier broader legacy optional-learning check stopped after 154 passes and
one failure in
`test_fifteen_fixed_paired_runs_are_audited_without_false_completion`: its report
lacked `execution_replay` for some failed evaluations (`run did not complete`).
This check is not recorded as passed, and the complete legacy optional-learning
suite has not been accepted by this change. See also the
[v3 integration record](../v3-integration-2026-09-27.md).

CUDA admission is exercised with resource fixtures. Real CUDA, H20/CARC,
multi-GPU throughput and Linux hardware acceptance remain unverified. Local
single-node CPU execution is the demonstrated hardware path. Office mode uses
CPU/RAM reserve proxies; it does not measure application frame rate or guarantee
desktop responsiveness. Recommendations are the best valid measured candidates
for the frozen batch under observed constraints, not a global optimum or a
scientific hyperparameter search.

Local ignored evidence is indexed at
`artifacts/ray-tune-engineering/2026-09-27/`. It retains test reports, metadata,
calibration measurements and task-branch history. Large binary checkpoints stay
in the disposable original test directories; the archived metadata is not a
standalone replay/model package. This evidence is engineering-only and is not a
formal experiment manifest from integrated `main`.
