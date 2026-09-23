# Replay debug prototype (SO-MARL grid factory)

> **Status: prototype, not integrated.** This was written against the older
> `SmartSOM-2026-Spring` repository before this repository was known to the
> author, and is pushed here for discussion. It does **not** read `smartsom`
> simulator output: the only traces it can show today come from its own mock
> factory. It overlaps with the existing replay surfaces in
> `src/smartsom/studio/` (`playback.py`, `replay_clock.py`, `replay_dashboard.py`,
> `replay_inspector.py`, `replay_workspace.py`) and `src/smartsom/trace/`.
> Decide what to keep before building further: the browser viewer and the
> conflict / loaded-path metrics are the parts the Studio may not cover.

Step through an SO-MARL episode tick by tick on the factory grid: AGV paths,
conflicts, buffer nominations, machine modes, inspections, and every agent's
decision and reward at each tick. It targets the failure modes in the SOM
slides of 2026-09-15 (1,180 AGV conflicts per episode, loaded-path ratio 1.42),
which Gantt charts cannot explain.

```
writer ──▶ manifest.json + replay.jsonl ──▶ validate / metrics / render ──▶ replay.html
```

The writer only **writes** traces and the viewer only **reads** them. The viewer
is one static HTML file: no Qt, no Ray, no checkpoint and no server.

## Quick start (from the repository root)

```bash
# 1. Generate synthetic traces (mock factory; nothing else writes this format yet)
uv run python -m scripts.replay_debug mock --controller rule --seed 7 --html
uv run python -m scripts.replay_debug mock --controller marl --seed 7 --html

# 2. Check a trace against the schema
uv run python -m scripts.replay_debug validate runs/replay_debug/mock_marl_seed7

# 3. Slide-4 metrics table (one row per trace)
uv run python -m scripts.replay_debug metrics runs/replay_debug/mock_*_seed7

# 4. Build a self-contained viewer (trace embedded; share this one file)
uv run python -m scripts.replay_debug render <trace_dir> -o replay.html
```

Traces are generated evidence and go to `runs/replay_debug/`, which is gitignored.

Tests: `uv run pytest tests/unit/test_replay_debug.py -q`.

You can also open `viewer.html` directly in a browser and load `manifest.json`
and `replay.jsonl` with **Open trace…** (or by drag and drop).

## Viewer

| Area | What it shows |
|---|---|
| Grid | Slide-11 layout. Machines have a status border (grey idle, blue processing, orange blocked, red breakdown), a mode letter (S/N/F) and a progress bar. Buffer slots show job numbers, with a dashed gold border on the job that buffer nominated. Inspection slots show `?`, and a dashed purple border while inspecting. AGVs have a coloured trail, a dashed goal ring and a red ring when in conflict. A red ✕ marks the cell of a movement conflict. A green or red dot on a job shows a PASS or FAIL inspection; a red dot on the other corner marks a rush order. |
| Timeline | One lane each for AGV conflicts, pickup/drop-off conflicts and defects, deliveries, scrapped jobs and inspections. Click to jump. |
| Controls | ⏮ ◀◀ ◀ ▶ ▶▶, speed, trail length. ◀◀ / ▶▶ jump to the previous or next event of the selected type. |
| Tick tab | Counters; the selected entity's state and return; events this tick; every decision this tick (action, top-3 probabilities, local reward). The selected entity's decisions are sorted first. |
| Events tab | Every event, filterable by type. Click a row to jump. |
| Job tab | One job's attributes (due date, estimated vs true defect status) and full history. |
| Summary tab | Slide-4 metrics, local return by agent type, and the worst loaded trips (click to jump). |

Keys: `Space` play/pause · `←` / `→` step (`Shift` = 10) · `Home` / `End` · `n` / `p` next/previous event.
Click an AGV, machine, buffer or inspection station to select it. Click a job square to follow that job.

## Trace format `smartsom.somarl.replay.v1`

A trace is a directory with two files. The format was defined against the old
repository's `smartsom.trace.v1` exporter and does **not** yet match
`src/smartsom/trace/records.py` in this repository; aligning the two is an open
question for review.
A single bundled `{"manifest": ..., "frames": [...]}` JSON file is also accepted.

### `manifest.json`: static, written once

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `"smartsom.somarl.replay.v1"` | required |
| `trace_id`, `episode_id` | str / int | |
| `controller` | str | e.g. `rule`, `somarl`, `somarl+sl` |
| `seed`, `horizon` | int | |
| `layout` | object | `width`, `height`, `obstacles: [[x,y]]`, `stations: [...]`, `operations`, `modes` |
| `layout.stations[]` | object | `id`, `type` ∈ machine / pre_buffer / post_buffer / input / output / inspection / disposal / charger, `cell: [x,y]`, `interaction_points: [[x,y]]`, `capacity`; machines add `operation`, `pre_buffer`, `post_buffer` |
| `agents[]` | object | `id`, `type` ∈ machine / buffer / quality / dispatcher / mover, `entity` |
| `jobs[]` | object | `id`, `release_tick`, `due_tick`, `rush` |
| `synthetic`, `decision_probs` | optional | mock traces set `synthetic: true`, `decision_probs: "synthetic"` |
| `reward_config` | optional | reward constants used |

Coordinates are `[x, y]`, with x = column, y = row and the origin at the top left.

### `replay.jsonl`: one JSON object per tick, ticks consecutive

| Field | Content |
|---|---|
| `tick` | int, increases by exactly 1 per line |
| `agvs[]` | `id`, `pos: [x,y]`, `status` ∈ idle / moving / servicing / waiting / charging, `carrying` (job id or null), `goal` ([x,y] or null), `task` (`{job, pickup, dest, phase}` or null), `battery` (optional) |
| `machines[]` | `id`, `status` ∈ idle / processing / blocked / breakdown, `job`, `mode` ∈ SLOW / NORMAL / FAST, `remaining`, `total`, `operation` |
| `buffers[]` | `id` (a station id), `slots: [job or null]`, `nominated` (job or null) |
| `inspection[]` | `id`, `status` ∈ idle / inspecting, `remaining`, `slots: [{job, state ∈ not_inspected / inspecting / inspected} or null]` |
| `jobs{}` | active jobs only: `location` (station or AGV id), `next_task` (O1–O4 / DELIVER), `done_ops`, `quality` ∈ unknown / pass / fail, `p_defect`, `defective` (ground truth), `due_tick`, `rush` |
| `sinks{}` | optional cumulative counts per output / disposal station |
| `decisions[]` | `agent`, `agent_type`, `action` (a **semantic label** such as `START(J3,FAST)` or `LOAD@M1.POST`, not a slot index), `mask` (allowed labels), `probs: [[label, p], ...]` or null |
| `rewards` | `shared` (factory reward this tick), `local: {agent_id: r}` |
| `events[]` | see below |

State fields describe the factory **after** the tick is applied. An AGV that
picks up or drops off does not move in the same tick, so its `pos` equals the
service cell.

### Events

| `type` | Required fields |
|---|---|
| `job_released` | `job` |
| `dispatch` | `agv`, `kind` (load / unload / charge / wait), `station`, optional `job` |
| `pickup`, `dropoff` | `agv`, `job`, `station`, optional `cell` |
| `service_conflict` | `agv`, `station`, `reason`, optional `job` (a pickup/drop-off conflict) |
| `movement_conflict` | `agv`, `kind` (`agv` / `env`), `cell`, optional `other` |
| `op_start` | `machine`, `job`, `mode` |
| `op_done` | `machine`, `job`, `defect_introduced` |
| `machine_blocked` | `machine`, `job` |
| `inspection_start` | `station`, `jobs` |
| `inspection_done` | `station`, `results: {job: pass or fail}` |
| `delivered` | `job`, `observed_quality`, `defective`, `lateness`, `on_time`, optional `due_tick` |
| `scrapped` | `job`, `station` |

### Metrics (computed identically in `metrics.py` and the viewer)

- **Total / passed / defect throughput:** `delivered` events, split by true `defective`. Passing rate = passed / total.
- **Total AGV conflicts** = movement conflicts (`movement_conflict`, AGV–AGV + AGV–environment) + pickup/drop-off conflicts (`service_conflict`).
- **Loaded-path ratio:** mean over loaded trips (`pickup` → `dropoff` for the same AGV) of cells moved / shortest road distance (slide 10).

## Mock factory (`mock_episode.py`)

A small seeded simulator that writes valid traces, so the viewer, schema and
metrics can be developed before the research simulator emits traces. **Its
numbers are not research results.**

- `rule`: reserves destination capacity, never moves into an occupied cell, runs machines in NORMAL mode (FAST when a job is short on time), and inspects jobs with an estimated defect probability above 0.4. Over seeds 0–9: 0 conflicts and a mean path ratio of 1.03.
- `marl`: 12% random moves, no reservations, 50% FAST mode, random inspection choices. It stands in for an untrained policy; over seeds 0–9 it averages 1,000–2,100 conflicts and a path ratio of 1.55. Its action probabilities are synthetic.

Rewards follow slides 18–33. The late-delivery penalty for 50–100 ticks late is
not given on the slides; the mock uses the 100–200 bracket (−75%).

## Open questions before this is integrated

1. Does `src/smartsom/studio/` already cover this? If so, keep only the gaps
   (conflict counts, loaded-path ratio, a shareable browser view) and drop the rest.
2. If the format survives review, reconcile it with `src/smartsom/trace/records.py`
   and with the factory map templates in `src/smartsom/studio/templates/`, rather
   than keeping two trace vocabularies.
3. Only then write frames from the simulator: per tick, the state after the tick,
   the tick's decisions (semantic labels, masks, policy probabilities), rewards and
   events; `validate`, then `render`.

Not implemented: side-by-side comparison of two traces on the same seed; any
reader for `smartsom` simulator output.
