# SmartSOM

SmartSOM is a manufacturing simulator for scheduling research. It models machines,
AGVs, finite buffers and disruptions, with rule-based scheduling, optional PPO
training, live visualization and recorded replay.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
to install the locked environment once. The `studio` extra enables the viewer.

```sh
git clone https://github.com/jian2028/SmartSOM.git
cd SmartSOM
uv sync --locked --extra studio
source .venv/bin/activate
```

Run the commands below from the repository root. In each new terminal, activate
with `source .venv/bin/activate` again. On Windows PowerShell, use
`.venv\Scripts\Activate.ps1`. You only need `uv` again when installing dependencies.

## Try a factory and replay it

Start with Template 1: a symmetric layout with four machines, four AGVs and
12 orders following four different processing routes. This uses a scheduling
rule and needs no trained model.

```sh
smartsom run --config configs/runs/template1_static.yaml --render-mode human --verbose
```

The window shows production as it runs; the terminal reports progress and the
recording directory. The verified reference completes all 12 orders at tick 288.
To reopen your recording, replace `RUN_DIRECTORY` with the printed path:

```sh
smartsom playback RUN_DIRECTORY
```

Playback starts paused at 0.25× (0.4 seconds per tick). Play/Pause, steps and the
timeline control recorded time; paused animations freeze and seeking restores an
exact integer frame. Machine bars lose one segment per processing tick, gears
hold the active job, and the lower band highlights the recorded speed mode.
Job badges show public quality only (`?`, `✓`, `×`); processing masks a previous
PASS as `?`, and a magnifier marks jobs currently being inspected.

The replay workspace uses a top playback timeline and KPI strip, a large factory
canvas with a dedicated right-side object inspector, and a resizable, collapsible
bottom analysis area. Click anywhere on its arrow strip to collapse it and
restore its previous height. Run Analysis opens on Overview; segmented tabs give
direct access to Output, Resources, Orders and Events. Resources separates machine
mode statistics from Orders/inventory, and Events lists the latest 200 recorded
events through the displayed tick. Double-click an event to seek to its frame.
Changing views does not change recorded time, playback, selection or zoom.

The State inspector groups public information by task: machine order and remaining
processing ticks, AGV position/load/battery and execution feedback, buffer inventory,
and inspection batches and statistics. Its segmented countdown shares the factory
machine renderer. Design and Raw frame remain separate. Below 1,200 pixels the
inspector becomes an overlay drawer, opened by selecting a resource or the map's
Inspector control and closed with its close control or Escape. Entering this compact
layout initially collapses analysis to preserve map space; it can be expanded again.

Qualified deliveries, output passing rate and rolling throughput remain in one
global strip above the canvas. The throughput calculation window (20/100/500 ticks,
default 100) and chart display range (last 100/500 ticks or all history through now)
are independently selectable. Both trend charts show their current value; hover
for exact tick values or click to pin a readout. No future statistical samples are
shown. The status bar shows playback state and zoom. A rounded handle at the middle of the map’s left edge expands or collapses Resources,
following the sidebar edge when open.
The four-corner icon in Map tools fits the entire map; More → Presentation contains the explicitly labeled charging preview.

Counts distinguish resident jobs from cumulative delivery/disposal. Input and output buffers share centered badge, stacked-job and count rows. Input shows outside `Wait` separately from resident inventory; its full-width bottom bar counts down recorded new-demand arrivals in integer tick segments, using the same segment renderer as machines. At each arrival it resets to the next recorded interval; no further recorded arrival leaves a neutral bar. Output buffers show qualified total, cumulative passing rate and rolling throughput in place. The machine analysis bars count each successful processing start once, regardless of processing duration. Percentages appear inside segments only when the fixed-size label fits with padding; hover any machine row for all mode percentages, counts and total starts. All statistics stop at the displayed integer tick. The map-edge outside queue card uses recorded
future arrivals, explicitly labeled **From recording**, rather than predicting
agent-visible arrivals. Click a job or resource for its state; Resources toggles
the optional sidebar. Design and raw frame details remain available in the
inspector. Charging preview is a labeled visual demonstration, not recorded
energy behavior. These presentation changes do not edit the factory or recording.

Try the same factory with four extra orders arriving during execution and one
scheduled machine outage:

```sh
smartsom run --config configs/runs/template1_disturbed.yaml --render-mode human --verbose
```

The reference delivers all 16 orders by tick 403 and runs to its configured
2,000-tick horizon. See the [case and verification guide](docs/template1-replay.md)
for routes, event times and independent checks. Omit `--render-mode human` for a
terminal-only run. Recordings are generated locally, not included in the clone.

## Optional: train and evaluate

Install learning dependencies once, then use the same activated environment:

```sh
uv sync --locked --extra studio --extra learning-marl --extra cpu --extra tensorboard --inexact
smartsom doctor --preset marl_micro
smartsom train-evaluate --preset marl_micro --name first_marl
```

This is a small MARL engineering example, separate from the Template 1 demo.
Training completion does not guarantee that a model completes every order or
outperforms the rule. Template 1 currently has no accepted learned controller.
To evaluate a model you trained, use the experiment directory printed above:

```sh
smartsom evaluate RUN_DIRECTORY --checkpoint last --baseline spt --render-mode human
```

[Python examples](examples/README.md) cover training and evaluation scripts.
The [terminal and monitoring guide](docs/runtime-display.md) describes shared Rich
progress, summary/debug controls, file logs and the read-only `smartsom monitor`.
The [experiment guide](docs/experiments.md) covers configuration, other learning
backends, checkpoints and resume.

## Where to go next

| Goal | Start here |
| --- | --- |
| Change the factory, orders or disturbances | [Scenario authoring](docs/scenario-quickstart.md) |
| Edit a factory visually | Run `smartsom studio`; [Studio guide](docs/studio.md) |
| Understand live display, recordings and audit | [Runtime guide](docs/production-runtime.md) |
| Run or adapt Python examples | [Examples](examples/README.md) |
| Find validation and maintenance scripts | [Scripts](scripts/README.md) |
| Understand the code and contribute | [Architecture](docs/architecture.md), [contributing](CONTRIBUTING.md) |

Source is in `src/smartsom/`, configuration in `configs/`, reference inputs in
`data/`, and automated checks in `tests/`. Generated `runs/` and `artifacts/`
remain local. CP-SAT grid execution, battery behavior and video export are not
currently supported. Historical validation results remain tied to the source
versions recorded in their reports.

Studio also supports manually drawn frame states. Run `uv run --no-sync smartsom studio`,
open a factory, select **Edit → State…**, then configure jobs, machine/inspection
progress, AGV positions and buffer statistics. **Preview changes** is temporary;
**Apply** is undoable. Save preserves the drawing in the factory YAML and Export map
includes it in PNG/SVG. These are illustration annotations, not simulation initial conditions.
