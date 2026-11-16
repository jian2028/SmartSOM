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
   manual run executes the union of all owners and supported Ubuntu Linux/macOS
   platform slices. Native Windows is outside this CI maintenance scope.
   High-impact or uncertain PR changes also select full verification.

Current tuning/public owners execute RLlib/Tune PPO/DQN and lifecycle contracts.
Retained native-internal regression tests are historical implementation checks,
not evidence of RLlib migration. Retired phase/lifecycle tests have an explicit
[contract mapping](../archive/rllib-transition-tests/README.md). Real Ray pause and continuation, frozen
case receipts, ownership and cleanup checks remain distinct from mocked control
unit tests. No wall-time completion guarantee is made.

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

- `fast`: lean core, configuration, CLI and native scheduler contracts
- `gym`: minimal centralized Gym adapter
- `learning`: native extensions, train/eval and complete checkpoint lifecycle
- `tuning`: resident RLlib/Tune resources, search, continuation and actor failure safety
- `resource-learning`: Base/SL native PPO and DQN state equivalence
- `usability`: real offline W&B and TensorBoard event/RNG acceptance
- `studio`: offscreen Qt authoring and current read-only playback
- `learning-public`: scientific calibration and real whole-run CLI control
- `control`, `macos-learning`, `diagnostics`: explicit cross-platform repetitions of shared control, checkpoint and native learning contracts.

The `macos-learning` slice additionally runs representative official PPO and DQN
SL-attention sampling, updates and checkpoint continuation, plus the Morris
public CLI, fixed panels and best/last recovery tests. These are CPU functional
checks, not GPU or long-training performance acceptance. The Morris fixture
reserves two logical CPUs for its driver and evaluation worker.

All native macOS slices use ARM `macos-latest` (three CPUs, 7 GB RAM), which
supports the locked Torch stack. Intel installation fails because Torch 2.14.0
does not supply a macOS x86_64 wheel. Dependencies are not downgraded to change
that outcome. See [GitHub's hardware reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

Heavy `macos-learning`, Linux `learning-public` and `tuning` use
`--isolate-selectors`: each complete manifest selector
executes serially in a fresh pytest process. Framework/model state retained by
earlier test modules therefore cannot remain in the pytest parent of later public
CLI acceptance. Real admission guards, budgets and strict deadlines stay intact.
Every scope uses the same required dependencies and exact skip policy. The runner
combines all coverage and JUnit, rejects missing reports/selectors, unexpected
skips and duplicate node IDs, and continues later scopes after a failure while
retaining that failure. It never treats a smaller selected subset as full coverage.
Later ARM failure snapshots were polluted by orphan workers and do not establish
hardware as the cause of every delay. Hosted ARM success still requires a fresh
full CI run; local Apple Silicon evidence is reported separately.

Native owners can use `--qualify-host` before pytest. The CI-only sampler requires
three consecutive five-second samples and has a ten-minute limit. Its retained
CPU and memory headroom thresholds qualify the test host; they do not alter Tune
resources or implement an application resource guard. Qualification failure is a
failure, not a skip or a test retry.

The Tune migration replaces broker/lease and online resizing acceptance with
stock-controller cohorts, simultaneous resident evaluation, full checkpoint
restore, whole-run pause, nested worker cleanup and Optuna state tests. Their
small scientific workloads establish correctness only. The separate development
performance gate requires fixed workloads, alternating repetitions and both
steady and end-to-end comparisons. Historical native scheduler timing evidence
does not establish acceptance of this implementation.

Public command observers with an inherited PID/birth journal scan the host at
four Hz, retaining independent child-generation registration and an immediate
scan plus predicate recheck at each successful boundary. Commands without that
journal retain their original observation frequency. Emergency cleanup uses the
unchanged exact-identity selector and launch gate.

Failed test fixture directories are recorded in owner coverage, including isolated
scope aggregation. Diagnostic staging visits those directories first and prioritizes
their public stdout/stderr, observer/deadline summaries and progress snapshots.
The existing 128-file, 256-KiB/file, 8-MiB payload and 20,000-entry scan limits remain;
checkpoint blobs and symlinks are excluded. Oversized stdout/stderr retain only a
bounded tail, labelled with original size and offset in the staging manifest.
Installing PySide6 and using offscreen Qt establishes automated UI contracts, not
real-window interaction acceptance. Linux path emulation is not native macOS
qualification. Native Windows is no longer maintained or qualified by this CI;
pre-existing Windows compatibility code and tests remain unchanged. Portable tests
still execute on supported platforms, with only exact, documented unsupported-OS
skip allowances. Keeping historical Windows cases is not a claim that they execute
on Linux or that Windows compatibility has been verified. CPU smoke tests do not
establish CUDA/H20 performance. Correctness tests do not establish unmodified hardware throughput.

## Local reproduction

Run resource-owning tests on a host with the declared CPU and memory capacity.
The real Tune phase tests declare three CPUs, including coordinator and nested
workers; insufficient resources must not be bypassed by changing scientific
inputs or suppressing required tests. Do not run performance comparisons beside
other test suites.

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

Replace `tuning` with an owner listed above. Use `--platform macos-latest` only
on actual macOS, and run its matching owners there. Ubuntu Linux, including an
Ubuntu environment inside WSL2, uses `--platform ubuntu-latest`. Native Windows
Python is not an execution target for this CI.
A full verification means running every planned owner/platform combination, not
just generating a full plan or running one suite. A minimal bare `pytest` run is
useful development feedback but is not equivalent to installed optional coverage.
Tests run serially: do not add `pytest-xdist -n auto` to suites that own Ray actors,
thread budgets, process state or native checkpoints.

### Experiments on Windows hardware

For future runs on a Windows PC, use Ubuntu inside WSL2 and a Linux Python 3.12
virtual environment created there. Do not invoke Windows Python, `python.exe`, or
a virtual environment created by Windows. Before syncing or running an experiment,
verify the active interpreter from the Ubuntu shell:

```sh
python -c 'import platform, sys; print(sys.platform, platform.system(), sys.executable); assert sys.platform == "linux"; assert platform.system() == "Linux"; assert not sys.executable.lower().endswith(".exe")'
```

Create/use the project virtual environment with Linux `uv` inside Ubuntu, then
repeat the interpreter check with `uv run --no-sync python` before execution.
Use the Ubuntu CI commands above. This documents the execution route; it does not
install WSL2 or establish that a particular Windows PC is ready. Formal experiments
still require the source identity and evidence described in `CONTRIBUTING.md`.

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

Existing Linux/macOS displayed checks are retained, including the Gym, fixed
learning, Ray Tune, resource, usability, CP, two `control (OS)` checks and two
`Native diagnostics and provenance (OS)` checks. Native Windows matrix entries
and the `windows-learning` job are removed. The resource-protocol check delegates to `resource-learning`, where those tests execute once. The legacy
`Python 3.12 CP acceptance` name is a compatibility check for the fast owner's
unsupported-grid-CP contract; it does not claim native solver acceptance.

The former `Portable foreground control` workflow is consolidated into `CI` so
one plan and aggregate covers the supported Linux/macOS platforms. Their displayed
check names remain stable, but workflow-specific rules or external dashboards may
identify the old workflow file. Historical required Windows checks may remain
pending and block merging after their jobs are removed. Inspect required checks
and those integrations, and verify the first run's actual check contexts. Removing
a historical Windows requirement from branch protection or repository rules needs
separate owner approval; this change does not edit those settings or substitute a
successful no-op Windows check.

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

Failure diagnostics are staged as regular files before upload. Staging skips
symbolic links, including cyclic checkpoint links, and caps traversal at 20,000
entries, publication at 128 files, individual files at 256 KiB, and total payload
at 8 MiB. The staging manifest records copied paths, bytes and any limit reached;
checkpoint/replay trees are never uploaded directly as failure evidence.
