# SmartSOM

SmartSOM is a manufacturing simulator for scheduling research. One physical core
owns jobs, machines, AGVs, finite capacity, failures and time. Base negotiation
and centralized assignment use that core; native PPO and DQN support Base and
explicit social observations. Rule, random and Python policies use legal semantic
candidates. Engineering tests do not establish research superiority.

## Install and run

Use Python 3.12 and uv. Optional libraries are loaded only by their corresponding
execution paths; importing the simulator needs no Torch, Gymnasium, Ray or W&B.

```sh
uv sync --locked --extra learning --extra cpu --extra wandb
uv run smartsom check current-smoke
uv run smartsom run current-smoke
```

Daily scientific inputs remain Factory, Workload, Algorithm and Experiment.
`check` and `run` accept an explicit file, directory or unambiguous name under
`configs/runs`. `batch-run` uses the same implementation. Execution is foreground
by default. Select `--background` explicitly for detached server execution.

```sh
uv run smartsom check current-train --profile server
uv run smartsom run current-calibrate
uv run smartsom run current-train --profile server --background
uv run smartsom runs list
uv run smartsom attach RUN_ID
uv run smartsom pause RUN_ID
uv run smartsom resume RUN_ID
```

The current examples are small engineering recipes, not formal experiments.
Training and evaluation default to W&B online in project `SmartSOM`. Authenticate
outside configuration files, or explicitly select `--wandb offline` / `disabled`.
Local evidence always remains available. Offline segments can be uploaded later;
the run URL opens in a phone browser. TensorBoard recording is opt-in via
`--tensorboard`; start its viewer separately with `smartsom tensorboard RUN_ID`.

See the [current workflow](docs/current-workflow.md) for profiles, precedence,
advanced overrides, result filtering, configuration comparison and bash/zsh
completion. Profiles change operational settings only. Scientific changes require
a new run; continuation validates frozen inputs, code and complete checkpoints.

## Studio and replay

Install `--extra studio` for the Qt factory editor and recorded playback:

```sh
uv run smartsom studio
uv run smartsom playback REPLAY_DIRECTORY
```

Factory authoring is broader than the current simulation contract: unsupported
inspection/rework layouts fail explicitly before execution. Historical recordings
remain readable without executing their former policies.

## History and development

Legacy physics/configuration/framework runtimes are archived, not selectable by
current commands. Existing runs and artifacts remain in place. Consult the
[migration guide](docs/migration/current-runtime.md), [archive index](docs/archive/index.json)
and [current architecture](docs/architecture.md). Historical ADRs and measured
results retain their original scope and source identity.

Use the [CI ownership guide](docs/ci.md) for validation. Formal experiments must
start from integrated `main` or an explicit tag. This consolidation does not launch
formal experiments or migrate old model weights automatically.
