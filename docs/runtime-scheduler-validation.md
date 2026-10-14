# Unified scheduler cloud engineering validation

## Final resource model and Rich controls acceptance (2026-10-14)

This completion extends the local `b177bdfcdc5b5569bc108ccc9f7fb732b009fb5f`
checkout. Earlier sections below are historical attempts, not the final-source
acceptance. The source/test/configuration manifest contains 750 files with aggregate
SHA256 `842a8302ea7b403541f39d1444cb88793eecc744d156e78b9ae9d08150894de9`.
Tests ran on this frozen worktree; subsequent documentation and the local commit
are mapped back to these exact source bytes rather than described as post-commit tests.

| Final6 gate | Result |
| --- | --- |
| Complete minimal fast, lint/format and inventory | 2129 passed, 2 skipped in 514.74s (0:08:34) |
| Affected resource/Tune unit files | 203 passed in 57.47s |
| Real scheduler, control, PPO/DQN lifecycle and native continuation | 58 passed; 2 calibration cases deselected and run separately; 566.68s |
| Real dual-mode validation, cross-evaluation and resume, both retention policies | 2 passed in 27.60s |
| PPO fixed and DQN adaptive calibration, execution and completed resume | 2 passed; 3 non-calibration cases deselected; 133.88s |
| Rich rendered fixtures | 27 cases: 1/4/12 tasks at 60/80/160 columns, normal/unknown/training-complete with workflow remaining |
| Real CLI PTYs | 21 cases: 9 read-only, 9 exact-target removal/cancel/stale rejection, 3 selected-task pause/resume |
| Optimizer counterexamples | 3 widths: a known training budget does not imply optimizer progress or speed |
| Actual DQN observation | 2 captured normalized snapshots, 12 SVG/PNG views; recorded update4/32, real validation and learner diagnostics |

The PTYs exercise page selection, refresh, resize, details and exact stable task
identity. `P` pauses the selected task and `R` requests its continuation; Space
controls the whole batch. Read-only input never changes control bytes or mtimes.
No Rich per-task stop button was added. Global resume preserves individual holds,
and a paused worker releases its slot only after safe save and verified exit.
Real-worker tests cover replacement work, queued resume and resource headroom.

Resource admission consumes semantic-group stage observations and retains model
highwaters. A temporary uncalibrated startup envelope is bounded by both local
and shared capacity, then separately settled after a verified committed update
with a memory observation. Configured, measured and predicted floors remain.
Terminal publication retains estimates and peaks while marking current usage
inactive only after verified release. Independent review found no remaining
blocking findings in these changes.

Earlier failed attempts are retained: one checkpoint fault-injection race was
corrected without loosening its assertions; a PTY harness dictionary-order
assumption was corrected to revision order. Final4 native found a genuine
10%/20% reserve mismatch (1 failed, 32 passed before intentional interruption);
Final5 completed with 56 passed and 2 missing-terminal-DTO failures. Neither
attempt is counted as final acceptance. Final6 reruns the complete selected gates.
Completed fixture cache was offered for eviction without changing file contents;
Final5's expanded native artifacts were replaced by a byte-compared archive to
retain all evidence while freeing disk. Runtime budgets and deadlines were unchanged.

The complete supported Linux/macOS CI owner union remains unexecuted. Mac and
GPU hardware are outside this run. Sampled resource estimates are not OS resource
isolation or proof against transient peaks; controlled-resource fairness tests
do not establish a universal waiting-time bound. Synthetic layout fixtures are
not training observations. ClearML remains a future adapter contract. No push,
PR, deployment or old experiment restart is part of this completion.

## Historical acceptance records

This record covers the Linux cloud checkout on
`feat/unified-runtime-scheduler`, based on
`5734b6fde225c6de96bc9d648d99c765947921a5`. These are development engineering
checks, not formal scientific experiments, a release, or a performance claim.
No Mac execution, ClearML deployment, credentials, push, PR, or main merge is
part of this work.

The cloud host reports Linux x86_64, five logical CPUs and 17.59 GiB RAM. No X11
`DISPLAY` or Wayland display is configured. Rich recording and installed
librsvg/Cairo produced SVG/PNG exports; stdlib PTY drove real CLI input. Chromium
was present but its headless probe timed out. No screenshot software was installed.
These exports do not constitute desktop terminal screenshots.

## Scope and review

The change connects ordinary directory batches, Tune batches and independently
appended native work to persistent execution control and shared admission. The
public API, CLI and controlling Rich attachment distinguish received commands,
applied intent and observed worker state. Logical removal retains artifacts and
requires target-bound confirmation. Read-only monitoring never submits commands.

Review covered command ordering/idempotency, revision checks, target identity,
safe worker exit, shared slot and CPU/RAM accounting, default drain versus
keep-open, manual holds, task removal, and display state. Review found and fixed
double reservations for native children, paused status aggregation, launch-failure
cleanup, atomic append retries, and oversized historical memory estimates blocking
unrelated work. Final follow-up checks cover task namespace collisions, explicit
parent ownership, external-pressure fairness and idempotent CLI removal retries.

## Executed evidence

Final runtime acceptance completed **47 passed, 2 deselected in 519.52 seconds**
after namespace, fairness and CLI corrections. The later UI-only notice wrapping
fix was verified separately by the full fast gate and PTY checks below.

```sh
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-sync pytest \
  tests/integration/test_runtime_scheduler.py \
  tests/unit/test_runtime_control.py \
  tests/integration/test_tuning_batch_runtime.py \
  tests/integration/test_tuning_native_continuation.py \
  -k 'not calibrates' --basetemp=/tmp/smartsom-native-verified -vv
```

The two calibration cases were subsequently run separately: **2 passed,
3 deselected in 126.37 seconds**, covering real PPO fixed and DQN adaptive
calibration, execution, checkpoint identity and completed-run resume:

```sh
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-sync pytest \
  tests/integration/test_tuning_batch_runtime.py -k calibrates \
  --basetemp=/workspace/smartsom-calibration-recheck -vv
```

Together these cover all cases in the selected native files, not the complete
tuning CI owner. The lifecycle assertions include:

- Real independent spawned counter workers: 4 to 6 without restarting the
  original PIDs, then target 2 with all six initially retained and four exiting
  naturally before replacement admission. Persisted counters prove progress.
- A public batch CLI process: keep-open, append with frozen science, repeated
  append command after removal of its original authoring input, task controls,
  confirmed removal, and orderly close.
- Genuine native Tune PPO and DQN: pause after an actual training commit, verify
  the old actor exited and shared reservations cleared, preserve an individual
  hold through global resume, then continue in a new process to all 32 updates.
  The selected continuation checkpoint hash remains unchanged.
- Separate PPO/DQN subprocesses: compare the restored state exactly with the
  saved state and the resumed final state exactly with continuous training under
  the same seeded environment. This includes optimizer/RNG and algorithm-specific
  collector/simulator/replay state; the bounded case runs 4 then 8 physical ticks.
- Existing native validation/evaluation, frozen-partner, sampler-thread and
  remaining-budget continuation regressions.

Tune broker, Ray adapter, resources, batch and dashboard focused tests completed
**292 passed in 66.87 seconds**. The scheduler/author-driver focused combination
completed **207 passed in 60.58 seconds** after final runtime review corrections.
Final UI regression checks completed **145 passed in 12.30 seconds**:

```sh
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-sync pytest \
  tests/unit/test_tuning_broker.py tests/unit/test_tuning_ray.py \
  tests/unit/test_tuning_resources.py tests/unit/test_tuning_batch.py \
  tests/unit/test_tuning_dashboard.py tests/unit/test_workflow_dashboard.py -q
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-sync pytest \
  tests/unit/test_runtime_control.py tests/unit/test_runtime_display.py \
  tests/unit/test_tuning_broker.py tests/integration/test_runtime_scheduler.py \
  tests/integration/test_author_driver.py -q
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-sync pytest \
  tests/unit/test_task_pages.py tests/unit/test_runtime_display.py -q
```

The final complete Linux fast gate passed: **2075 passed, 2 allowed Windows-only
skips in 431.96 seconds**. Coverage reports no missing selectors, unexpected skips,
deselection or failures. Ruff lint/format and `git diff --check` passed.

## Visual and interactive acceptance

Real renderer exports cover main, details and resource-blocked fixture states at
80×24, 120×36 and 160×50. Actual PPO/DQN running snapshots and completed parent
states are archived separately with capture timestamps and observed status.
These are individually atomic JSON observations, not a cross-file transaction.

The PTY harness ran real `smartsom attach` and `monitor` processes, captured raw
terminal bytes and intercepted only render output for SVG recording. Its scheduling
state is a labelled fixture, not a running learner. Six scenarios passed: readonly,
cancel, narrow cancel, wide confirmation, narrow confirmation and long-ID narrow
confirmation. Escape, wrong IDs and repeated D/d submit no command; correct typed
confirmation submits exactly one target-bound command. The harness verifies a
received receipt, not worker exit; real process lifecycle tests establish exit.
Resize covers 120×36 → 80×24 → 160×50.

Visual testing discovered that the old 80-column notice clipped the required ID.
Notices now wrap, and the prompt separates batch, task, effect and confirmation.
80-column screenshots and tests verify the entire target, short ID, Enter and Esc,
including long names, within a 23-row render. The narrow footer can still abbreviate
late shortcut hints with an ellipsis; confirmation data is never abbreviated.

Artifacts, scripts, raw PTY output, PNG/SVG images and machine-readable checks are
in `/workspace/smartsom-runtime-evidence/`. Start with `README.md`,
`checks-after-wrap.json` and `images/removal-80.png`. These generated files are
delivered separately from the Git source change. The image titles distinguish
fixture state from recorded actual training. No desktop GUI or Mac font/key
compatibility is claimed. EOF behavior has unit coverage; the PTY bundle does not
claim a real network-disconnect test.

## Resolved verification failures

The initial full fast runner was invoked with a temporary fixture root inside the
repository, causing five nested pytest accounting failures. Moving `RUNNER_TEMP`
outside the repository fixed that invocation problem. The next full run reported
2061 passed, four stale-fixture failures and two allowed Windows-only skips. Three
fixtures lacked the newly read Tune plan; one expected a slot without its new
CPU/RAM charge. The fixtures now assert the authored initial limit and complete
charged reservation; their focused rerun completed **58 passed**. Neither failed
full run is counted as a passing gate.

A later fast attempt encountered infrastructure I/O errors while `/tmp` was full
of retained test artifacts. Completed fixture directories were moved intact to
`/workspace`, with lookup symlinks preserving old evidence paths. The final gate
uses a fresh workspace temporary root and passed in full. An overlapping calibration
attempt also failed its final live-resource recommendation check despite three
valid measurements. That filtering code was unchanged; its report lacks enough
capacity detail to distinguish CPU from RAM pressure retrospectively. An isolated
rerun passed both calibration cases without changing guards, budgets or assertions.

## Reproduction and evidence boundaries

The lean gate uses its own locked environment so it does not remove native extras:

```sh
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-project python \
  scripts/ci/selection.py --event workflow_dispatch \
  --output /tmp/smartsom-ci-plan.json
RUNNER_TEMP=/workspace/smartsom-ci-delivery-final \
UV_PROJECT_ENVIRONMENT=/tmp/smartsom-fast-env \
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run --no-project python \
  scripts/ci/runner.py --suite fast --platform ubuntu-latest \
  --plan /tmp/smartsom-ci-plan.json --sync
```

Use fresh temporary directories when reproducing, with sufficient disk space.
The original native acceptance path now resolves to preserved artifacts under
`/workspace/smartsom-native-verified`. Final logs are copied into the separately
delivered evidence directory; they are not committed experiment artifacts.

Six counter processes are not six real PPO learners. The actual PPO/DQN lifecycle
proofs use one learner at a time. Native appended work adapts to resource headroom;
Tune retains measured throughput grouping. Logical budgets do not impose OS RSS
or CPU limits. Safe pause requires a supported checkpoint boundary and does not
promise arbitrary-instruction restore. Full native state acceptance here applies
to composable PPO/DQN, not every historical backend. Numerical equality on this
host is not cross-platform reproducibility.

The complete Linux/macOS CI owner matrix has not been executed by this cloud
task. Dependency/ownership changes conservatively select full CI; the focused
native proof and Linux fast gate do not substitute for that merge gate. Mac is
explicitly outside the user-authorized execution scope. ClearML has a public
adapter contract only, with no deployed integration.

See the [commands and rendered Rich comparison](unified-runtime-scheduler.md).
Those Rich examples are deterministic layout fixtures, not training observations.

## Independent delivery review and matrix repair

Independent read-only review of `adb92ab` found that a native matrix retained
its frozen per-file concurrency cap after a shared runtime limit increase. The
inner driver now uses the applied shared limit; standalone execution retains
its configured cap. Review also identified repeated admissions against one host
snapshot. Shared matrix workers now start at most once per observation and
resample before the next admission.

The repaired author-driver suite passed **21 tests in 93.37 seconds**, including
actual default workers with limit changes **1 → 2 → 1**, distinct live PIDs,
advancing durable receipts during draining, unchanged existing PIDs, and no third
worker admission. A separate test uses synthetic host readings with actual
workers to require distinct resource observations between launches. The runtime
control/display/Tune broker subset passed **175 tests in 4.99 seconds**. Ruff
lint and all 612 Python format checks passed.

The first new real-host test attempt was blocked by the existing memory reserve:
retained test files had filled the container's charged page cache. Offering the
completed fixture files' clean cache for eviction reduced cgroup usage from
about 14.2 GB to 5.2 GB without deleting or changing evidence. The successful
rerun retained real observations and unchanged resource guards.

The review explicitly does not establish a per-phase peak model: memory is a
per-task sampled high-water mark. Native admission does not reproduce Tune's
calibration-group throughput model. Fresh samples can lag startup peaks, and
separate matrix drivers do not atomically reserve observed host headroom.
Explicit shared budgets atomically limit declared resource contributions but
remain logical budgets, not operating-system isolation.

On the repaired source, the native runtime/continuation selection completed
**46 passed, 1 failed, 2 deselected in 415.74 seconds**. The one PPO lifecycle
failure was caused by editing this validation document during the run: its
new dirty path changed the Git status frozen before the replacement actor
started. The actor correctly rejected that checkout identity; implementation
and dependency checks had passed, and its checkpoint and released slots were
intact. With repository state held stable, the exact PPO case then passed
**1 test in 115.74 seconds**. This is a failed attempt followed by a successful
targeted rerun, not a single all-green invocation. Both PPO/DQN separate-process
full-state equality cases passed in the original selection.

Before/after manifests in the evidence bundle compare 748 tracked source,
test, configuration, dependency and ownership files: all bytes match throughout
these repair reruns. The tests ran before committing the repair; this manifest
mapping establishes the final source bytes rather than claiming a post-commit
test invocation. The historical complete fast gate and calibration results
above predate this repair; the full supported CI owner union remains unexecuted.
