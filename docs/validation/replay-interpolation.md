# Replay interpolation verification

This is local engineering evidence on `codex/replay-interpolation`, based on
`128ac4e` plus uncommitted changes. It does not change experiment results or
promote a research milestone. The local evidence directory is
`artifacts/replay-interpolation/`; generated evidence is not included in a clone.

## Behavior

Offline playback uses a monotonic presentation clock and a 16 ms Qt refresh
interval. At 1×, one recorded tick still takes 100 ms. Position and remaining
processing/inspection segments interpolate; discrete job ownership, quality,
arrivals and delivery counters stay at the preceding committed record until the
next boundary. The details pane always identifies that integer tick.

Pause freezes the fractional position. At 10.4, backward goes to 10 and forward
goes to 11; integer positions step by one. Seeking pauses. Changing speed keeps
the current phase. Playback stops at the exact last frame. Maximum may omit
visible frames; it does not interpolate a straight path across skipped ticks.
Adjacent records are cached, and details update only at integer boundaries.

Live playback and the simulator are unchanged. Historical frames are read through
a display adapter, not converted to executable current-runtime traces. Historical
quality markers, pool counts, job annotations and original metrics are retained;
raw historical job details replace the obsolete player's separate table.

## Automated checks

- 36 focused tests passed: presentation clock, pause/resume, speeds, endpoints,
  forward/backward steps, cache reads, discrete details/cargo, machine outage and
  final processing tick, inspection progress, segmented drawing, and historical
  malformed/empty/truncated recordings and original factory identity.
- Broader regression: 1,518 passed and 124 were deselected by
  `-m 'not learning and not marl'`. Three additional tests failed in the sandbox
  because Ray/W&B could not bind local sockets or write their service cache.
  All three passed on a local rerun outside the sandbox, with W&B explicitly
  offline. This is not a claim that all learning acceptance tests were rerun.
- Ruff check, formatting check and `git diff --check` passed.
- Every integer endpoint and every half-tick AGV position of the following four
  recordings was checked, along with cargo and integer detail timestamps.
  Original recording hashes remain unchanged. See `final-recording-checks.json`.

| Recording | Last tick | Qualified deliveries |
| --- | ---: | ---: |
| Current Template 1 static | 288 | 12 |
| Current Template 1 disturbed | 2,000 | 16 |
| Historical rule, seed 24004 | 5,000 | 167 |
| Historical MARL, seed 24004 | 5,000 | 121 |

The disturbed run's delivery completion and its 2,000-tick horizon remain distinct.
Historical runs still use their original continuous-arrival scenario.

## Native macOS checks

All four recordings were opened in actual Qt windows. Play/pause, forward/backward
integer steps, timeline jumps, mouse dragging, all speed choices and terminal
states were exercised. Examples captured include static tick 34→35 at 40%,
disturbed tick 49→50 at 41%, and the 167/121 historical endpoints. Disturbed
19→20 and 39→40 boundaries were inspected through native controls. The automated
outage test separately covers processing paused by downtime.

Screenshots and timestamped accessibility observations are in the local evidence
directory: `static-pause.png`, `disturbed-pause.png`, `rule-initial.png`,
`static-end.png`, `disturbed-end.png`, `rule-end.png`, `marl-end.png`, and
`native-actions.json`. Initial computer-control mouse errors were resolved by
reselecting/raising the window; mouse dragging was subsequently verified.

Intermediate positions and frozen progress were visible during native checks.
The 16 ms timer is a refresh target, not a measured guarantee of sustained 60 FPS.
No cross-platform or video-export acceptance is claimed.

## Open the existing local recordings

```sh
source .venv/bin/activate
smartsom playback artifacts/template1/static-replay
smartsom playback artifacts/template1/disturbed-replay
python scripts/validation/historical_replay.py play rule
python scripts/validation/historical_replay.py play marl
```

The first two aliases require the existing local Template 1 evidence. From a
fresh clone, generate recordings using the commands in the README. Historical
commands require the separate bundle described in
[historical replay](../historical-replay.md).

## Retention and source identity

The historical `play` entrypoint now uses the current viewer. It no longer imports
or exposes the frozen old player as an alternate viewing path. Original frames,
models, episode reports and frozen execution source remain evidence.

`cleanup-inventory.json` records local retention decisions. One unreferenced early
one-order warmup recording is a deletion candidate, pending explicit user
confirmation. An older static recording remains because `native-ui.json` refers
to it. Validation, training and failure-diagnosis evidence is retained; existing
static/disturbed aliases are preserved. No recording has been deleted as part of
this change without confirmation.

`source-manifest.json`, `workspace.patch` and `changed-source.tar.gz` identify the
uncommitted implementation. The branch remains uncommitted and unpushed. The two
previous README/CI commits were separately pushed to `main` at the user's request.
