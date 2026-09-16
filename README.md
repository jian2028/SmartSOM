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

Playback starts paused. Use Play/Pause, forward/backward steps, the timeline and
Speed. The detail panel lists jobs and buffer contents.

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
