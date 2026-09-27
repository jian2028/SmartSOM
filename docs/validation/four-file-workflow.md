# Four-file authoring and execution engineering verification

Verification dates: 2026-09-27–28. Checkout: uncommitted `main` based on
`80a28decdaa4ccc7802c31b48ce50d80ff027a1f`. This record covers engineering behavior,
not a formal research experiment or trained-policy performance claim.

## Implemented surfaces

- Factory v2 reliability defaults/entity overrides and Studio round-trip preservation.
- Workload v3 same-pool Low/Mid/High ordering, measured V, normal/rush allowances,
  independent data streams and exact rational processing reference work.
- Algorithm v2 rules/central/resource Agent contracts and detached Experiment v4
  parsing, typed overrides and real native matrix tasks.
- Named/versioned custom rules, public semantic actions/features, finite JSON state,
  explicit extension modules and drift rejection before worker imports.
- macOS detached launch handshake, owned-process stopping, native phase ledgers,
  complete-update training recovery and evaluation-stage restart.
- Existing calibration/Tune routing, recommend/auto, committed phase continuation,
  real task progress and outer-run source evaluation.

See [daily workflow](../four-file-workflow.md), [student rules](../student-rules.md),
[command lifecycle](../command-workflow.md) and [ADR 0025](../decisions/0025-four-file-authoring-and-background.md).

## Completed checks

Commands use `uv run --no-sync`, explicit root `PYTHONPATH`, a disposable uv cache
and existing environments; no optional package was installed for this acceptance.
All generated runs are below `/private/tmp/`, outside the project's result folders.

| Check | Fresh completed result | Evidence directory |
|---|---|---|
| Complete base pytest suite, stable source | 1785 passed, 274 skipped; exit 0; 282.48 s | `/private/tmp/smartsom-four-config-base-final2-20260927` |
| New config/rule/background/native batch plus legacy Study regression | 110 passed; exit 0; 55.05 s | `/private/tmp/smartsom-four-config-regression-20260927` |
| Optional CPU learning and unified safe-stop | 10 passed; exit 0; 155.14 s | `/private/tmp/smartsom-four-config-optional-final-20260927` |
| Frozen source evaluation and CLI/display compatibility | 99 passed, 4 skipped; exit 0; 5.02 s | `/private/tmp/smartsom-four-config-source-check-20260927` |

The optional checks include real SB3 centralized PPO, RLlib centralized PPO and
RLlib resource PPO: 16 physical ticks, two updates, two validations and final
evaluation. They verify changed weights, unchanged frozen partners, distinct data
and policy seeds, complete-update recovery, evaluation-only restart and completed
stage skipping. Unified stop checks also exercise parallel Study admission/drain,
owned calibration cancellation and committed Tune trial recovery.

Native/background checks exercise real registered rule processes, four-entry
matrix concurrency, terminal-input closure, launcher exit, startup failure,
stop/repeat-stop/background resume, and saved monitor rendering at narrow/wide
widths. JSON output and noninteractive logs contain no interactive ANSI progress.
PID reuse, timeout/force and unrelated-process preservation are tested separately.

## Final affected checks

The complete base suite above preceded the last Tune snapshot compatibility and
structured probe-deadline fixes. Fresh affected checks after those fixes:

| Check | Completed result | Evidence directory |
|---|---|---|
| Calibration/probe, source evaluation, CLI/display, native/background | 193 passed; exit 0; 37.16 s | `/private/tmp/smartsom-four-config-final-focused-20260927` |
| Real central/resource CPU learning, calibration cancellation and Tune stop/recovery | 7 passed; exit 0; 122.99 s | `/private/tmp/smartsom-four-config-optional-final2-20260927` |
| Real recommend, auto and short-deadline Tune pipeline | 3 passed; exit 0; 52.31 s | `/private/tmp/smartsom-author-tune-20260927-accept3` |
| CLI evaluation from a completed outer Tune author run | completed; exit 0; 1.671 s; one actual case | `/private/tmp/smartsom-author-tune-20260927-accept3/test_real_v4_tune_calibration_1/evidence/source-evaluation-verified.json` |

Source evaluation retained the exact frozen data, left the author/Tune/training
records byte-identical, and did not create another training attempt. Exact owned
process inventories were empty; the test-owned Ray processes had exited before
emergency cleanup.

The last presentation-only terminal-summary synchronization also passed fresh:
92 config/Tune display tests (exit 0, 3.16 s), and three real Tune tests (exit 0,
47.43 s) under `/private/tmp/smartsom-author-tune-20260927-accept4`.
The final monitor snapshots show recommend/recommended with zero training ticks,
auto/completed with 16 ticks, and short calibration/failed with an unstarted queued
entry. No completed entry retains a stale running summary. All three registered
process inventories are empty; the auto run's Ray leftover list was empty before
emergency cleanup.

Final Ruff lint, formatting (454 Python files), offline lock check (121 packages)
and `git diff --check` passed with exit 0.

Final-code source evaluation also passed from accept4 (exit 0, completed, 1.348 s,
one case): exact frozen inputs, unchanged training/author/Tune records and one
training attempt. Its evidence is
`/private/tmp/smartsom-author-tune-20260927-accept4/test_real_v4_tune_calibration_1/evidence/source-evaluation-verified.json`.
Qt Studio editor/template acceptance using the preserved current `.venv` and
offscreen Qt completed with 65 passed, exit 0, 5.86 s, under
`/private/tmp/smartsom-four-config-studio-final-20260928`.
The final CLI help-text clarification is presentation only; run/resume help in
the current main environment exits 0 and names the v4 override owners explicitly.
Its final config/unified-entry regression completed with 73 passed, exit 0,
3.71 s, in `/private/tmp/smartsom-four-config-final-help2-20260928`.
Lifecycle acceptance requires readable kernel process metadata: the managed
sandbox denied `ps` in an initial invocation; these completed lifecycle results
use the approved environment that can verify PID creation times.

## Protection checks

- HEAD is unchanged and no commit, merge or push was performed.
- Large SHA-256 remains
  `ee68146af3d91d065fa2e4537147e364af3c5243fb565d6a87be275816ff48f5`.
- Built-in Template 7/8/9 file digests match the initial checkout copies.
- The existing composable-policies worktree's saved 100-file snapshot has no
  changed/missing files. Other pre-existing worktrees were only read; where no
  byte baseline exists, unchanged bytes are not claimed.
- The modelling Markdown was read as authority and not edited by this task.
  It changed externally during this work; relevant H/V and due-date sections were
  refreshed rather than treating an old hash as current authority.
- Three task-owned detached implementation worktrees remain under `/private/tmp/`.
  Their dirty changes are preserved; they are not forcibly removed.

## Scope and limits

macOS is the actually tested platform. POSIX Linux background code is implemented
but was not run on Linux. Windows detached launch explicitly rejects the request.
GPU, H20, CARC, shared/remote Ray clusters and multi-node execution are not qualified.

Real calibration acceptance bounds candidate enumeration to a one-thread,
one-entry profile and a test-owned two-CPU Ray runtime. It verifies the real
probe/learner/scheduler path, not the globally best resource allocation. A deadline
may return a feasible measured baseline without convergence; this is explicitly
recorded and cannot be described as a proven optimum. A missing baseline rejects
automatic launch. Calibration does not measure operating-system GUI smoothness,
search scientific hyperparameters or automatically alter network contracts.

V generation rejects data when its finite candidate rules cannot establish strict
Low < Mid < High. Large 2,000-Job pool generation cost and formal 18×3 research
matrices have not been benchmarked or created. No student/first Small input file
or formal training run was created. Social Learning remains an extension interface,
without experience sharing or distillation.

Training resumes only at saved complete updates. An interrupted evaluation stage
is restarted; within-case continuation and saving an incomplete forced update are
not promised. New source evaluation preserves frozen data; adding cases or
replications requires a new Experiment. Older v2/v3 runs retain their previous
compatibility/recovery limits and are not migrated.
