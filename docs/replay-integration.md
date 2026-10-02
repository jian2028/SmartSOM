# Integrated desktop Replay

Replay composes the Studio `FactoryScene`, `FactoryView`, resource items, job
symbols, `FactoryStateLayer`, property tree and workspace chrome. It does not
maintain a separate copy of the map or simulator. Templates 10, 11 and 12 are
included exactly as authored in main; they contain no chargers.

## Interaction

The global KPI strip and playback controls stay available above the workspace.
The resizable analysis pane belongs below the map; Timeline is its default tab,
with Overview, Output, Resources, Orders and Events retained. The resource list
starts collapsed. Interaction-point squares are hidden by default in both Studio
and Replay; their Layers menu can show them. Destination rings remain visible
and fit just inside one grid cell. The inspector shows global metrics until an object is selected.
Click blank map space or Global performance to clear selection; Escape also
closes the compact inspector drawer. Below 1200 pixels, that drawer overlays the
map but not the playback controls. More → Raw frame remains available globally.

The timeline has movement conflict, pickup/drop conflict, qualified delivery,
scrap, inspection and machine outage tracks. Counts refer to the recording or
selected object, not just elapsed time. Click a marker to pause at its result
frame and list events; hover shows its evidence. A wheel zooms about the pointer,
plus/minus zoom about the playhead, Home fits the recording, and arrow keys step.
Pixel aggregation expands as the time scale is enlarged. Outages show intervals
only when both transitions are recorded. Previous/next marker controls use the
chosen event category and object filter, with no wraparound.

AGV colors are deterministic by semantic identity. A recorded conflict adds a
red ring without replacing the identity ring; no exclamation badge is added. Other rejected actions
remain distinct in execution feedback/events. Dashed circles show recorded
active destination ports, labelled only with AGV IDs.
Coincident destinations share a label listing AGVs. Only the selected AGV has a
history trail (off, 12, 30 or 60 ticks); matrix trips never acquire a fabricated
grid route. Existing processing, inspection and job symbols remain shared with
Studio. N/F/S badges indicate recorded machine mode; Replay cannot edit it.

## Evidence and compatibility

`replay_model.py` adapts current and historical rows for display, without changing
trace files. Timeline coordinates are result frames; the decision inspector
separately identifies each recorded decision boundary and result frame. Phase
order is preserved. Orders link to decisions containing that exact job identity.

Only complete, identifiable legacy PPO logits become probability bars. DQN
scores remain Q values. Unknown scores remain explicitly unspecified scores.
A selected action or single log probability never becomes a full distribution.
Current composable PPO/DQN recordings typically omit full distributions and say
“Full probabilities not recorded”. No model is loaded to regenerate them.

Task performance is computed in `trace/performance.py`, shared by Replay and
evaluation summaries. Recent throughput states its window; tardiness requires
completion identities and due ticks. Missing values show “Not recorded”.
Theoretical references carry their nominal processing assumptions and exclusions
in the global inspector; they are not controller targets or empirical results.
The theme changes shared window chrome and leaves semantic map colors consistent.

## Local verification

The integration worktree contains generated engineering evidence under
`artifacts/replay-integration/` (ignored by Git): rule recordings on the unchanged
three templates, light/dark screenshots, compact drawer views, and results of
opening existing PPO/DQN recordings. These are UI acceptance evidence, not formal
experiment results. `validation.md` in that directory records the local sources,
checks and limitations. The historical-format integration tests use preserved
fixture semantics and test read-only access; no archived historical run was
available at the documented local archive path during this task.

Run focused tests with the project's `uv` environment:

```sh
QT_QPA_PLATFORM=offscreen uv run pytest -q \
  tests/unit/test_replay_model.py tests/unit/test_task_performance.py \
  tests/unit/test_performance_rows.py tests/unit/test_replay_evidence.py \
  tests/integration/test_production_playback.py \
  tests/integration/test_historical_playback.py \
  tests/integration/test_task_performance_ui.py
uv run ruff check .
```

### Replay presentation audit fixes (2026-10-02)

Inspection statistics now scroll independently, and custom chart text follows the
workspace palette. The standard analysis pane is more compact; reference assumptions
are expandable. Timeline counts explicitly cover the entire recording (and selected
object when filtered); inspection counts represent events, not jobs. The inspector
labels retained AGV feedback separately from the current execution result. Missing
DQN action values are labelled Q values, without converting them to probabilities.
The performance panel distinguishes unrecorded evidence, no submissions, no elapsed
ticks, and an inapplicable passing-rate reference. Simulator and metric arithmetic
are unchanged. GUI checks use recorded evidence; they are not new experiments.

### Startup and map navigation (2026-10-02)

Playback startup shows a terminal square spinner with validation percentage and
named configuration/statistics/index/window stages, plus a native loading dialog.
The grid reader optionally retains up to 32 MiB of serialized validated state/event
summaries; startup analysis reuses these rather than decoding full decision payloads
again. Full frame reads still validate hashes, and uncached summaries fall back to
normal reads. Historical readers retain their existing path.

The 1 GiB Template 12 recording measured 24.85 s before and 8.08 s after under the
same cProfile harness; window construction fell from 16.80 s to 0.38 s. These are
local engineering measurements, not a general latency guarantee.

Mouse wheels, pixel-delta scrolling and native macOS pinch events zoom around the
pointer. The map has a fixed upper-right fit button, current-view clipboard capture,
and a toggle for interaction points. Capture excludes the floating controls and
preserves replay time and view state. Real hardware pinch remains a manual check;
Qt native gesture delivery is covered by regression tests.
