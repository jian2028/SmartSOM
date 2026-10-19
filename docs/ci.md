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

This first rollout uses conservative whole-file native owners. In particular,
`learning` still includes the unchanged three-backend 16,384-step acceptance
and 32,768-step multi-resource update contract. Affected learning checks can
therefore remain long. Full currently means the complete supported Linux/macOS
owner/platform union, not that every expensive case has already moved into a
full-only shard. A finer
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
- `learning-public`: complete configurable batch/recovery files, each in a fresh
  serial pytest process on a separate Ubuntu runner
- `tuning`: calibration, PPO/DQN continuation, batch/online lifecycle and failure
  safety, including both controlled-clock and real-clock online lifecycle tests
- `resource-learning`: sparse resource protocol and native resource learning
- `usability`: public lifecycle, reporting, search and tracking, including report
  coverage formerly activated incidentally by CP dependencies and Optuna parent
  display coverage; Node is also required for existing script contracts
- `studio`: offscreen Qt editor, rendering and playback acceptance
- `control`: native Linux/macOS filesystem/control/display contracts; its
  dependencies include genuine display learning tests, so this is not the
  minimal-import boundary
- `macos-learning`: native checkpoint/long-path/Unicode restore, plus supported
  macOS learning and process contracts
- `diagnostics`: native diagnostic/provenance contracts on Linux/macOS

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

Tuning additionally qualifies the host before pytest, using the unchanged
performance-mode resource broker and requiring three consecutive five-second
samples with at least 2.25 available CPUs after the application's reserve and
external load. This accounts for two one-thread native learners plus observer
headroom. Qualification has a ten-minute bound and fails closed. The main-run
38351590310 failure recorded 22 pressure park/release cycles and no observed
two-running-learner overlap, despite genuine commits; its raw capacity repeatedly
fell below two CPUs. Selector isolation releases prior framework state and host
qualification avoids starting this acceptance with insufficient capacity.
Neither changes application admission, the real 240-second/controlled
180-second guards, learner budgets or required overlap/restore/cleanup proofs.
Passing the repaired hosted run remains the acceptance gate.

The separate native Tune pause/hold/resume acceptance keeps its 32 real updates,
checkpoint identity checks and two genuine actor pauses. Its final resume now
uses a 360-second hard limit and fails after 60 seconds without a new durable
batch commit. Two unchanged local DQN repetitions took 236.95 and 235.53 seconds
from that resume to normal exit, leaving only 3–4 seconds under the former
240-second wait. Observed commit gaps stayed below 11 seconds; final evaluation
and exit also fit well inside the stall limit. These measurements establish a
narrow acceptance margin, not the exact phase of the earlier hosted timeouts.
The second repetition initially waited for RAM; its final-resume measurement
excludes that initial admission delay. No resource guard or training input changes.
Restored counters are baselined before resume. Only increasing integer update
counts, bounded by the frozen training budget and matching physical ticks, reset
the stall clock; timestamps, stage changes and replayed counters cannot extend it.
The bounded `deadline-summary.json` records commits and final native/admission
observations before cooperative failure cleanup. Normal exit still must satisfy
all completion, checkpoint, progress and resource-release assertions. The separate
online lifecycle 240/180-second guards remain unchanged.

The macOS learning owner additionally uses `--qualify-host` after dependencies/preflight
and before pytest execution. It reads the unchanged balanced broker capacity and
requires three consecutive five-second samples with at least 1.25 free CPUs and
256 MiB above its normal RAM reserve. This qualifies room for one training thread
plus the measured test observer; it does not alter the application's reserve or
ignore aggregate CPU. An unprimed CPU sample is never sufficient. Qualification
has a ten-minute bound and logs every observed input; timeout fails the owner,
never skips tests or turns missing execution into success. The configurable public checkpoint/stop deadlines remain 180/60 seconds. Its
fresh resume uses a monotonic 480-second hard limit and fails after 120 seconds
without new bounded training/update/validation/evaluation counters. Replayed
counters, stage/status changes, resource samples and timestamps cannot refresh
that watchdog; each validation round has its own tick maximum. The interrupted
snapshot is baselined before waiting. Scientific budgets and completion/cleanup
assertions are unchanged. This is deadline qualification, not a throughput claim. No test retry, runner service termination,
dependency downgrade or resource reading override is used.

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
Online lifecycle fixtures additionally publish a rolling cached-observation
snapshot (at most256 KiB, normally once per second), retaining bounded commit,
park/release/restore tails, raw admission budgets, lease/owner generations and
overlap/stop state before Tune returns. Deadline and owned-exit records survive
emergency cleanup. These files, restore journals and stop requests are included in
the existing bounded diagnostic whitelist; the real240-second and controlled
180-second guards, host monitor, CPU/RAM reserves and acceptance assertions remain
unchanged. Synthetic snapshot-writer tests are not live concurrency evidence.

Installing PySide6 and using offscreen Qt establishes automated UI contracts, not
real-window interaction acceptance. Linux path emulation is not native macOS
qualification. Native Windows is no longer maintained or qualified by this CI;
pre-existing Windows compatibility code and tests remain unchanged. Portable tests
still execute on supported platforms, with only exact, documented unsupported-OS
skip allowances. Keeping historical Windows cases is not a claim that they execute
on Linux or that Windows compatibility has been verified. CPU smoke tests do not
establish CUDA/H20 performance. The bounded real-clock test checks elapsed-time lifecycle qualification; neither it nor the
controlled-clock companion measures unmodified hardware throughput.

## Local reproduction

The fast owner's real native matrix concurrency test requires a host with at
least four CPUs. This public repository's `ubuntu-latest` job supplies four CPUs
and 16 GB RAM under [GitHub's standard runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Two-CPU private-repository runners are not sufficient for this real parallel
acceptance; provision adequate capacity rather than bypassing admission or skipping it.
Before starting workers it uses the same fail-closed performance
host qualifier as Tune, requiring three consecutive five-second samples with
2.25 free CPUs after the broker reserve and 1 GiB above the unchanged shared
native memory reserve (at least 2 GiB or 20% of host RAM). A host exposing fewer than 3.25 CPUs
fails immediately; temporary pressure has a 180-second qualification bound.
Insufficient capacity is a failure, never a skip or a changed admission reading.
The test retains 500 real evaluation cases and checks live worker identities,
new durable receipts and draining without restarting workers. Dispatch,
steady-state progress and cleanup each retain 45-second bounds; initialization
and first receipts have a separate 180-second bound because each worker validates
and freezes the full case inventory before publishing its first receipt.

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
