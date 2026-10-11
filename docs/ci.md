# Tiered CI and test ownership

The `CI` workflow uses an explicit ownership manifest, not a guess that every
`unit` test is inexpensive. Test selection is an engineering verification gate;
it does not promote a roadmap milestone or establish research performance.

## What runs automatically

1. **Fast, always:** a minimal locked environment, Ruff lint/format, dependency
   boundaries, deterministic core/configuration and public rules-only contracts.
2. **Affected native:** pull requests add the native dependency and platform
   suites selected by changed paths and changed test ownership. Real training,
   restore, public entrypoint, process cleanup and online lifecycle assertions
   stay in their affected PR suites.
3. **Full:** every push to `main`, daily at approximately 04:23 UTC, and every
   manual run executes the union of all owners and supported platform slices.
   High-impact or uncertain PR changes also select full verification.

This first rollout uses conservative whole-file native owners. In particular,
`learning` still includes the unchanged three-backend 16,384-step acceptance
and 32,768-step multi-resource update contract. Affected learning checks can
therefore remain long. Full currently means the complete owner/platform union,
not that every expensive case has already moved into a full-only shard. A finer
node-level cost split must retain genuine bounded native counterparts and a
verified impact/dependency map; no budget is shortened to achieve a time target.
There is no 20-minute completion guarantee. Native pytest output names individual
tests so a long acceptance case can be distinguished from missing progress.

To request full verification, open **Actions → CI → Run workflow**, select the
branch, and run. There is no fast-only manual override that can substitute for
required native checks. GitHub schedules run from the default branch and may be
delayed by service load. New pushes cancel older runs for the same workflow/ref;
a cancelled run is not accepted as successful evidence.

The plan job fetches complete history, computes the merge-base diff against the
PR base, and tests the same checked-out merge SHA used by the test jobs. It prints
why each suite is selected and saves `ci-plan.json`. Changed tests select their
owners even if no production path changed. Unknown paths, unavailable/invalid
diffs, renamed/deleted tests, shared fixtures/helpers, workflow or selector
changes, dependencies and shared core/scientific contracts conservatively expand
to full. Invalid or incomplete ownership is an error, not a reason to omit tests.
Documentation-only changes still run the fast gate.

## Owners and intentional repetition

`.github/ci/test-ownership.json` is the executable source of truth. Each test file has
one primary owner; explicit repetitions preserve native OS checks, minimal
optional-dependency boundaries, and the small existing Linux control/diagnostics
check contexts. The [inventory](ci-coverage-inventory.md) quantifies these exceptions. Secondary platform slices can select exact functions.
The owners are:

- `fast`: minimal imports and deterministic contracts, including unsupported
  grid CP entrypoint rejection
- `gym`: the deliberately minimal Gym environment and core/Gym equivalence
- `learning`: native learning, public API and learning state contracts
- `tuning`: calibration, PPO/DQN continuation, batch/online lifecycle and failure
  safety, including both controlled-clock and real-clock online lifecycle tests
- `resource-learning`: sparse resource protocol and native resource learning
- `usability`: public lifecycle, reporting, search and tracking, including report
  coverage formerly activated incidentally by CP dependencies and Optuna parent
  display coverage; Node is also required for existing script contracts
- `studio`: offscreen Qt editor, rendering and playback acceptance
- `control`: native Linux/macOS/Windows filesystem/control/display contracts,
  including Windows publication; its dependencies include genuine display
  learning tests, so this is not the minimal-import boundary
- `windows-learning` and `macos-learning`: native checkpoint/long-path/Unicode
  restore, plus the platform's supported learning and process contracts
- `diagnostics`: native diagnostic/provenance contracts on Linux/macOS/Windows

Installing PySide6 and using offscreen Qt establishes automated UI contracts, not
real-window interaction acceptance. Linux path emulation is not Windows/macOS
qualification. CPU smoke tests do not establish CUDA/H20 performance. The bounded
real-clock test checks elapsed-time lifecycle qualification; neither it nor the
controlled-clock companion measures unmodified hardware throughput.

## Local reproduction

Run from the repository root with the pinned Python 3.12 and uv 0.12.22. Planning
uses the standard library; execution syncs the exact locked extras for its owner.
`--sync` intentionally removes unrelated extras from the current project virtual
environment. Use a separate worktree/environment if preserving a different local
set of extras matters.

```sh
uv run --no-project python scripts/ci/selection.py --event pull_request --base origin/main --head HEAD --output ci-plan.json
uv run --no-project python scripts/ci/runner.py --suite fast --platform ubuntu-latest --plan ci-plan.json --sync
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

For offscreen Studio on Ubuntu, install the system Qt runtime libraries first:
`sudo apt-get install --no-install-recommends libegl1 libopengl0 libxkbcommon0`.
The selected Studio CI job installs these from the official Ubuntu repositories;
they are not Python project dependencies. Concrete QtGui/QtWidgets imports are
required before test collection.

For a full plan and a particular native owner:

```sh
uv run --no-project python scripts/ci/selection.py --event workflow_dispatch --head HEAD --output ci-plan.json
uv run --no-project python scripts/ci/runner.py --suite tuning --platform ubuntu-latest --plan ci-plan.json --sync
```

Replace `tuning` with an owner listed above. Use `--platform macos-latest` or
`--platform windows-latest` only on that actual OS, and run its matching owners there.
A full verification means running every planned owner/platform combination, not
just generating a full plan or running one suite. A minimal bare `pytest` run is
useful development feedback but is not equivalent to installed optional coverage.
Tests run serially: do not add `pytest-xdist -n auto` to suites that own Ray actors,
thread budgets, process state or native checkpoints.

## Adding or changing tests

Update `.github/ci/test-ownership.json` with every new test file. Primary owners
cover complete files, including every function added to a mixed file. Secondary
platform slices may select exact functions. Choose the real dependency owner,
platforms and impact domains; keep exactly one primary file owner and explain any
intentional environment/platform repetition. Do not assign a native test to fast
because its path contains `unit`, and do not replace real training or restore
assertions with mocks to meet a timing target.

Run the selector after the edit. Its inventory validation rejects unowned test
files, stale file selectors and ambiguous primary ownership. Pytest rejects stale
function selectors when the corresponding suite executes. Execute the affected
owner using the runner and inspect its selected, deselected, executed and skipped
counts. Optional dependencies required by an owner must import successfully;
unexpected collection or test skips fail the suite. A legitimate platform skip
needs a narrow, documented manifest allowance rather than a blanket skip waiver.
For selection or ownership changes, run the helper's focused regression tests and
full-plan inventory validation as well as the affected native suites.

## Required checks and evidence

No workflow-level path filter or job-level impact condition suppresses a required
check. Static test jobs always start, download the plan and either run their slice
or record an intentional `not_required` result. This is distinct from GitHub's skipped job
state. Matrix fail-fast is disabled so one platform failure does not erase other
platform evidence. Each lane publishes its result JSON; executing lanes also publish JUnit. Selected suites
must actually collect and execute their contracts.

`Python 3.12 quality gate` is now the always-running aggregate. It requires all
prerequisite jobs to succeed and verifies every expected owner/platform result
against the plan. Missing plans/results, failed or cancelled jobs, missing required
execution and unexpected skips fail closed. Lint/format live in the new fast job
and remain prerequisites of that aggregate.

Existing displayed checks are retained, including the Gym, fixed learning, Ray
Tune, resource, usability, CP, three `control (OS)` checks, `windows-learning`,
and three `Native diagnostics and provenance (OS)` checks. The resource-protocol
check delegates to `resource-learning`, where those tests execute once. The legacy
`Python 3.12 CP acceptance` name is a compatibility check for the fast owner's
unsupported-grid-CP contract; it does not claim native solver acceptance.

The former `Portable foreground control` workflow is consolidated into `CI` so
one plan and aggregate covers all platforms. Displayed check names remain stable,
but workflow-specific rules or external dashboards may identify the old workflow
file. Before enabling this revision, inspect those integrations and verify the
first run's actual check contexts. Updating branch protection or repository rules
is a separate owner-approved operation; this change does not modify them.

Artifacts are retained for seven days. Tuning preserves only the two bounded,
regular, non-symlink online lifecycle JSON reports (at most 256 KiB each), plus
limited failure diagnostics and JUnit. Learning failures preserve named regular
logs/reports. Checkpoint, replay and arbitrary temporary trees are never uploaded.
Timeouts bound jobs (30 minutes for fast/control, 45 for narrower native suites,
60 for broader suites); they are safety limits, not measured runtime promises.
Review actual CI duration distributions before tightening them. This change alone
does not claim a measured speedup or a completed cross-platform CI run.

### Shared helpers in test modules

Tests import helpers from other `test_*.py` modules across suite and platform
boundaries. Until a complete transitive helper dependency graph is maintained,
**every changed test module escalates to full verification**, including additions.
Direct owners are still listed as forced selections in the plan. This deliberately
chooses broader execution over silently missing downstream platform consumers.

The inventory matches pytest’s default `test_*.py` and `*_test.py` file discovery.
Changing `testpaths`, `python_files`, or adding an alternate pytest configuration
is rejected until the inventory logic is explicitly updated and regression tested.

Plan, runner and aggregate bootstrap through pinned uv/Python rather than the
runner host’s default Python. Stable plan/result artifact names are replaced on
a retried job, supporting partial reruns without losing successful owners;
attempt-numbered lifecycle and failure diagnostics remain separately retained.
