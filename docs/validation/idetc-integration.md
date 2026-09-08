# Frozen IDETC SPT integration acceptance

The question is whether the frozen IDETC inputs complete under current SmartSOM
semantics, replay exactly, and retain comparable evidence. This is not an exact
reproduction of the old simulator or paper's performance results. No learning,
PPO, SO-MARL, dynamic task-graph editor or transport CP is involved.

## Frozen source and conversion

`data/reference/idetc/source.json` records eight input byte hashes from old IDETC
commit `729bc692c85781be275f42144ce582449b875d98`. They match the frozen v6 lock
archived in commit `b668ba6e677e3410c5074a7c281d1c6004e0040a`; the lock stored at
729 itself was an earlier lock. The correct archived lock records 729 and dirty
state. The claim covers the verified blobs, not a completely clean historical
execution environment. `raw/` contains exact originals and the lock.

Draft4's 12 SPT rows were extracted directly from
`SmartSOM_IDETC_Draft4_2026-05-15.pdf` (SHA-256
`72023b1162f1b6fc43731191c78cdb45aa4b535e5b11d31b4026da79660bb66f`) and matched to
the local LaTeX table. That LaTeX file is untracked in the old repository; it has
a file hash, not a claimed Git version. All reference values remain contextual
columns, never acceptance thresholds.

`scripts/prepare_idetc.py` verifies originals before allocating an export directory.
It produces ordinary factory, instance, fixed arrival, scenario, algorithm and
study files. All four outputs are validated before writing; existing directories
are refused. No old runtime imports or dependencies are needed.

| Case | Jobs | Operations | Maximum release | Machines / AGVs |
|---|---:|---:|---:|---:|
| S00 | 100 | 283 | 149 | 8 / 4 |
| S01 | 100 | 291 | 284 | 8 / 4 |
| S10 | 100 | 283 | 149 | 8 / 4 |
| S11 | 100 | 291 | 284 | 8 / 4 |

Original job IDs, edited serial routes, releases, machine capabilities and all
121 directed matrix entries are preserved. An operation is `<job ID>/op_NNN`
(one-based serial index); a base mode ID is its eligible machine ID. Every mode
retains the old operation's duration. Reveals equal releases. Four AGVs start at
m-0 through m-3; pre/post buffers are 6 and b-hold is 999999. Quality modes are
M0=1.2/0.01, M1=1.0/0.018, M2=0.8/0.03 with public probability. MB/UPT are off.
The old quality threshold and volatility metadata do not become runtime rules.

## Fixed acceptance protocol

```sh
uv run smartsom plan configs/studies/idetc_spt.yaml
uv run python scripts/validate_idetc.py --workers 2
```

The committed study fixes seed 101, S00/S01/S10/S11, SPT-M0/M1/M2, replications
0–4 and the default control variant: exactly 60 runs. World inputs are shared per
case/replication; quality draws are paired across all three modes. Different cases
are different worlds. Entry/plan identities and all effective seeds are recorded.
Formal acceptance refuses uncommitted implementation/input changes or non-main
source; unrelated AGENTS.md/paper work is retained and recorded as dirty paths.

The normal batch coordinator calls run_one in up to two spawn workers, retains
failures, and continues. An interrupted batch can resume using saved snapshots;
failed attempts require explicit `--retry-failed`. Source/dependency changes
require a new study. Completed attempts have their evidence checksums verified.

```sh
uv run smartsom batch --resume PATH_TO_STUDY
uv run python scripts/validate_idetc.py --study-dir PATH_TO_STUDY --workers 2
```

After execution, each successful run is independently audited through:

1. Public action replay, checking the entire semantic trace, execution timetable,
   quality and makespan against recorded evidence.
2. Public full schedule replay, checking every physical/quality event and all
   processing, pickup, arrival and delivery times. WaitUntil may replace repeated
   WaitNextEvent, so unrecorded redundant wait actions are not reconstructed.
3. Public Simulator.run with a read-only recording policy for both action and
   schedule policies, checking every DecisionContext digest against the saved
   observation digest stream. Both policies use the existing step path.

Every replay keeps the kernel's per-transition invariants. Coverage independently
checks exactly one completion/quality decision per operation, one inspection and
output per job, final delivery makespan, and all quality summary fields. The audit
does not generate a replacement world, repair a schedule or write into run evidence.

Integration passes only if 60/60 complete and all 60 audits pass, with 20 paired
quality checks: M0 passed jobs contain M1, which contain M2. No overall makespan
monotonicity is required. Failed runs/audits remain explicit; no input, seed or
policy substitutions are permitted to obtain a passing count.

## Interpretation and reports

| Difference | Current treatment |
|---|---|
| Task-map edits | Already in frozen serial jobs; no runtime mutation |
| Dispatch and transport policy | Current processing-first SPT and shortest empty + loaded travel; no old proposal accept/reject loop |
| Holding | Old source's congestion fallback purpose; current reservations and precise unload rules |
| Buffers | Exclusive bookings, blocking, loaded waits, stable unload ordering |
| Quality randomness | One materialized draw per operation, shared across modes; old completion-ordered global draws are not replayed |
| Terminal time | Last actual output; old clock could return to last processing completion |
| Rounding | Half-up contract differs generally; all frozen durations × three factors have identical rounded integers here |

Study progress and each child's progress.log distinguish processing, output and
public inspection. Observation hashes are not full observation files. Full trace
and all realized inputs remain available; debug is off by default. Finite workers
and smaller recording files are not a claim of bounded-memory arbitrary-scale
execution or established training throughput.

Reports are local under `STUDY_DIR/acceptance/<unique audit>/`: report.json,
report.md, runs.csv, protocol.json, progress.log and per-run audits/. They include
source identity, actual dependencies, dirty paths, per-run metrics, group means,
sample standard deviations, success/audit counts, failure reasons and paper
reference columns. Statistics only include successful execution; audit counts
remain separate. Five replicates support this engineering observation, not a new
performance or statistical conclusion. Generated runs/reports are not committed.

## Verification status

Converter/audit focused tests validate source identity, route/matrix conversion,
rounding equivalence, pairing and portable export, plus deliberate corruption of
trace, schedule, summary, observation digests and file checksums. Holding's
independent 11-tick hand case and combination tests precede the full-size cases.
The full base and locked CP gates retain ft06=55, official FJSP=6 and Mk01=40.

Fresh macOS precommit checks: **17** focused converter/audit tests; base **840
passed / 12 optional CP skips**; locked CP **852 passed / zero skips**. Ruff,
format (145 files), lock consistency, both dependency environments and diff
checks pass. Logs and completed exit codes are under
`artifacts/idetc/integration-checks/`.

Development S00/M0 completed all 100 jobs and 283 operations (makespan 3069),
with 21 holding trips; both replay APIs and both observation audits passed.
Development runs are explicitly not the formal postcommit 60-run evidence.
The formal command must run after this protocol/input commit and its saved
report determines completion. Actual local verification uses macOS. Ubuntu CI
collects the same process, recovery, replay and regression tests; Linux remains
unverified until an actual completed CI run. This task does not push.
