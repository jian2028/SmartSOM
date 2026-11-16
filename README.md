# SmartSOM

SmartSOM is a manufacturing simulator for scheduling research. One physical core
owns jobs, machines, AGVs, finite capacity, failures and time. Base negotiation
and centralized assignment use that core. The active migration uses official
RLlib PPO/DQN and Tune for Base and implicit social-learning research models;
the supported core has local Linux engineering acceptance. Engineering tests do not
establish research superiority.

## Install and run

Use Python 3.12 and uv. Optional libraries are loaded only by their corresponding
execution paths; importing the simulator needs no Torch, Gymnasium, Ray or W&B.

```sh
uv sync --locked --extra learning --extra cpu --extra tuning --extra wandb
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
retain their historical components. The single maintained [RLlib campaign](experiments/current/README.md)
contains Small/Medium Base, SL-attention and SL-concat recipes, plus one composite
Rule baseline, internal Random fixed-panel recipe, and PR42 Morris/E1a-current PPO
per size (ADR0101). Old [V2](archive/legacy_runtime/experiments/v2/README.md)
and [ISL recipes](archive/legacy_runtime/experiments/implicit-sl/README.md) are archived.
`check` and `run` accept an explicit file, directory or unambiguous name under
`experiments`. Named Experiment YAML directories support
execution-only `--first-train-seeds N` selection, `--resume-missing` supplementation,
frozen manifests and `--id` / bound `--select` execution subsets
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
[RLlib recipes](experiments/current/README.md) for sharing, seeds and fixed panels.

For the supported core and deferred interfaces, see the
[migration status and configuration contract](docs/tune-migration.md).
`smartsom` remains the official entry point. Fixed configurations and Optuna
searches use the shared RLlib training and evaluation adapter. Configure total
logical resources and resident driver, Learner and EnvRunner requests in the launch profile.
Tune admits trials from those declarations; they are not measurements or OS limits.
Evaluation worker counts and the optional trial concurrency cap are set at launch.
The public PPO/DQN path uses FIFO and resident components. It no longer uses
ResourceChangingScheduler or a bootstrap actor. Official starting recipes are
`configs/algorithms/rllib-base-ppo.yaml` and `rllib-base-dqn.yaml`.
Core lifecycle and representative PPO/DQN performance checks passed locally;
deferred scientific interfaces receive explicit migration errors. Other platforms
remain unverified.

Whole-run `pause`, `resume`, `retry --all`, `stop` and confirmed `clean` remain.
Online `add/close/configure/remove`, task-level control and priority commands are
removed. Existing running experiments retain their archived source and commands;
this migration does not modify their processes, checkpoints or environments.

For single-task parallel sampling and PPO/DQN update schedules, see
[training controls](docs/training-sampling.md). Existing entry points are unchanged.

The [core closeout scope](docs/decisions/0098-rllib-core-closeout.md) retains rule
smoke and scientific calibration. Named Base Rule/Random fixed-panel evaluation is
also supported (ADR0100). Learning standalone evaluation, window/recovery analysis,
and new training recordings are deferred and rejected
by both `check` and `run` before allocating a run. Native learner/checkpoint formats
are historical inputs, not fallback execution paths.
