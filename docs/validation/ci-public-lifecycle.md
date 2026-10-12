# Public lifecycle CI repair qualification

This records engineering acceptance evidence, not simulator throughput or policy
quality. The baseline is merged PR10 commit `c6a10256a2ace0a60eb1cdd3837ec5fe28348eeb`.

## Trigger and evidence

[Postmerge main CI 38198714518](https://github.com/jian2028/SmartSOM/actions/runs/38198714518/job/114647112776)
collected 564 learning cases: 562 passed, while both configurable PPO/DQN
fresh-process resumes reached the 300-second limit before final publication.
Both completed 512 training ticks and all three nonuniform validation rounds;
final evaluation reached tick 339/344. The final observed 30.09/30.06 seconds
advanced 275/269 evaluation ticks. At that local slope, the remaining evaluation
would take approximately 19 seconds; publication and shutdown still require time.
This projection does not prove the cause of the broader job slowdown.

The failure archive scanned 17,185 entries and copied 128 files (760,188 bytes).
The file cap was exhausted before the failed public-test directories. Public
stdout/stderr were also absent from the allowlist. Failed fixture paths now flow
through coverage/isolated aggregation into bounded, prioritized diagnostic staging.

## Local baseline timing and limits

The unchanged DQN public stop/resume acceptance passed on this Mac before the
observer/watchdog changes: 1 passed, zero failures/skips, 152.12 seconds total.
The resume observer recorded 119.74 seconds, 60.29 parent CPU seconds, no remaining
owned identities and no pending/corrupt journal. PPO resume CLI/state reached
completed and its progress log spanned 115.77 seconds, but pytest/JUnit and final
cleanup evidence were absent after a transport interruption. PPO is not counted
as a passing local acceptance. Neither native path has been rerun on the repair
candidate while the Mac transport remains unstable.

The resume hard bound is 480 seconds: margin beyond the approximately 320-second
projected slow Linux resume plus publication/shutdown allowance. The separate
120-second no-progress bound is four display periods and more than twice the
approximately 56-second validation phases seen in the failed run. It also bounds
startup/admission with no semantic work. Restored training/update counters are baselined from the verified checkpoint
marker even if the initial progress file is missing, and only
bounded counters/ticks for the expected entry can refresh the monotonic watchdog.
Repeated/invalid counters, changed timestamps/status/resources, missing snapshots
and replayed validation rounds do not extend it. All counter maxima fit the
unchanged 512-tick/eight-update/three-validation/one-evaluation fixture.

The observer previously scanned the host every 20 ms during command waits. With
independent inherited identity registration, ordinary scans now occur every
250 ms and boundaries trigger a fresh scan and predicate recheck. This lowers
scheduled host-scan frequency by 12.5x; an actual CPU/runtime improvement has not
yet been measured. Nonjournalled observation and cleanup selection are unchanged.

## Required qualification

Pure tests cover idle and hard deadlines, heartbeat/replay rejection, validation
round resets, malformed evidence, failure propagation through the cleanup context,
scan cadence/fresh boundary checks, failed-fixture staging under file pressure,
log-tail bounds, symlink exclusion and missing/cancelled public-owner gate evidence.
These mocked cleanup-context tests do not replace native cleanup acceptance.

Before new regressions, collection proved the original 564 nodes partition exactly
into learning 494 + public 70. Nine new pure watchdog/observer cases are owned by
the existing complete public test file. Scientific budgets, assertions and all
other primary/platform selectors remain. The quality gate requires the new Linux
owner; both complete files remain in the isolated macOS owner.

Independent review and fresh full GitHub Ubuntu/macOS CI on the published final
head are required. Postmerge main CI must also pass before subsequent feature
publication. The previous merged-head success and local baseline DQN pass do not
qualify this repair candidate.
