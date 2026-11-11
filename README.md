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

Daily scientific inputs are Scenario, Algorithm and Experiment. Scenario combines
factory defaults, explicit machine positions/operations, job rules and breakdowns.
See [scene authoring and Studio](docs/scenario-authoring.md). The four standalone
[factory resources](src/smartsom/config/maps/) are shared by Studio and the installed
package; Scenario documents may embed their layouts with jobs and faults.
Algorithm components v2 explicitly selects learner/observation/encoder factories, per-stage
production policies, action permissions and a shared Mover. See
[algorithm components](docs/algorithm-components.md).
[Morris](configs/algorithms/morris.yaml) and [Justin](configs/algorithms/justin.yaml)
use the same components and entry points. The matched [V2 campaign](experiments/v2/README.md) groups shared/independent
Base/ISL, Morris, Justin and Random by map. Earlier
[ISL recipes](experiments/implicit-sl/README.md) remain available.
`check` and `run` accept an explicit file, directory or unambiguous name under
`experiments`. Named Experiment YAML directories support
execution-only `--first-train-seeds N` selection, `--resume-missing` supplementation,
append-only new members and ordered `--id` / bound `--select` execution subsets
([batch usage](docs/batch-incremental-usage.md)). Execution is foreground
by default. Select `--background` explicitly for detached server execution.

```sh
uv run smartsom check current-train --profile server
uv run smartsom run current-calibrate
uv run smartsom run current-train --profile server --background
uv run smartsom runs
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

Install `--extra studio` for the Qt scenario editor and recorded playback:

```sh
uv run smartsom studio
uv run smartsom playback REPLAY_DIRECTORY
```

Factory authoring is broader than the current simulation contract: unsupported
inspection/rework layouts fail explicitly before execution. Historical recordings require their archived source reader.

## History and development

Legacy physics/configuration/framework runtimes are archived, not selectable by
current commands. Existing runs and artifacts remain in place. Consult the
[recovery instructions](docs/archive/README.md), [archive index](docs/archive/index.json)
and [current architecture](docs/architecture.md). Historical ADRs and measured
results retain their original scope and source identity.

Use the [CI ownership guide](docs/ci.md) for validation. Formal experiments must
start from integrated `main` or an explicit tag. This consolidation does not launch
formal experiments or migrate old model weights automatically.

See [component ownership](docs/algorithm-components.md) and the current
[ISL study](experiments/implicit-sl/README.md) for sharing, seeds and fixed panels.

Native batch resources are updated together through one official command:

```sh
source .venv/bin/activate
smartsom configure RUN_DIRECTORY --max-concurrent 8 --evaluation-processes 5 --evaluation-cpus 5 --cpus 30
smartsom status RUN_DIRECTORY
smartsom status RUN_DIRECTORY --json
```

`--cpus` and `--memory-gib` are the whole-run budget. `--evaluation-cpus` is the
total CPU cap for one experiment's evaluation panel, shared by its workers; it is
not a per-worker request. Actual workers can be fewer than the target. Training
retains its baseline lease and shrinking evaluation drains current cases first.
These are cooperative admission quotas, not OS isolation. `--task-id ID` scopes
only evaluation settings; queue-wide settings cannot be combined with that scope.
All fields in a configure request validate and apply together. The returned receipt
acknowledges submission; `status --command-id ID --json` reports its application.

The CLI no longer accepts `batch-*`, `runs list/show`, `resume --retry-failed`, or
`stop --delete`. Use `run`, `configure`, `status`, `pause/resume/stop/retry`, and
`add/close/remove/clean`; see [the workflow](docs/current-workflow.md). Existing
running experiments retain their archived source and commands. No source or
checkpoint migration is performed.

For single-task parallel sampling and PPO/DQN update schedules, see
[training controls](docs/training-sampling.md). Existing entry points are unchanged.
