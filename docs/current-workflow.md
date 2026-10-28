# Current experiment workflow

SmartSOM uses four scientific input files: Factory, Workload, Algorithm and
Experiment. YAML selects parameters and explicit Python observation/network/policy
factories. The compiler freezes those files before execution. Historical inputs
and models use their original archived source, not a selectable legacy runtime.

## Install and inspect

Use Python 3.12 and the committed uv lockfile. Optional packages stay outside core
imports. For native learning and both monitoring sinks:

```sh
uv sync --locked --extra learning --extra cpu --extra wandb --extra tensorboard
uv run --no-sync smartsom check current-smoke --profile local
```

Add `--extra tuning` for Ray Tune and `--extra studio` for the desktop editor and
recorded playback. `check` never opens W&B or allocates an experiment directory.
It displays the resolved inputs, operation settings, origins and scientific hash.

## Launch choices

```sh
uv run --no-sync smartsom run current-smoke
uv run --no-sync smartsom run current-train --profile server --background
uv run --no-sync smartsom run path/to/experiment-directory --max-concurrent 4 --concurrency-mode adaptive
uv run --no-sync smartsom run current-train --wandb offline --tensorboard
```

These are alternatives, not commands to run repeatedly into the same output path.
Training and evaluation require a completed calibration record. The bounded
engineering examples form this explicit sequence:

```sh
uv run --no-sync smartsom run current-calibrate
uv run --no-sync smartsom run current-train
uv run --no-sync smartsom run current-evaluate
```

The example budgets are engineering checks, not research recommendations. Change
the four files for formal experiments and launch those only from integrated main
or an explicit source tag. Existing output directories are never overwritten.

An existing path wins over a short name. Short names search `configs/runs/` by
file stem or directory name; ambiguity fails with candidate paths. Directory runs
read direct Experiment YAML members and preserve their declared stage order.
`batch-run` is an alias of the same launch implementation. A directory containing
calibration/train/evaluate templates does not automatically resolve dependencies;
use a deliberately authored batch with available calibration inputs.

Profiles live in `configs/profiles/` or an explicit path. They are optional and do
not inherit other profiles. Priority is defaults, Experiment operation defaults,
explicit profile, explicit CLI. For example:

```yaml
launch:
  mode: foreground
concurrency:
  mode: adaptive
  max_concurrent: 4
resources:
  cpus: 8
  memory_gib: 12
tracking:
  mode: online
  project: SmartSOM
  tensorboard: false
```

Both supplied profiles default to foreground. `--foreground` and `--background`
are mutually exclusive. Profiles cannot change seed, learning rate, observation,
reward, train/validation budgets, sampling process count or device. Advanced
`--set algorithm.learning_rate=...` remains available for a new experiment and
is recorded explicitly. Resume uses frozen scientific inputs, never edited author
YAML. Concurrency limits independent trials; it does not change rollout size or
learning updates. Resource budgets are admission ceilings, not OS hard quotas.

Native PPO/DQN use Torch eval mode for rollout, bootstrap, optimizer forwards and
evaluation. Optimizer forwards still enable gradients and update trainable weights.
This keeps PPO's recorded and recomputed probabilities on the same distribution
and prevents DQN target inference from changing state. Dropout is disabled and
BatchNorm running statistics stay fixed; DQN target refresh copies the complete
state, including buffers. Custom networks must support singleton requests and
deterministic, batch-independent eval forwards. BatchNorm without running
statistics (`track_running_stats=False`) does not meet that contract. Action
sampling and DQN epsilon exploration remain explicit seeded policy operations.

## Find, control and compare

```sh
smartsom runs list
smartsom runs show RUN_ID
smartsom status RUN_ID
smartsom attach RUN_ID
smartsom pause RUN_ID
smartsom resume RUN_ID
smartsom stop RUN_ID
smartsom results runs --algorithm ppo --seed 101 --status completed
smartsom config diff current-train RUN_ID
```

Commands accept a run directory or unique ID prefix. Supply `--root PATH` for
custom output roots; multiple roots can be provided. No mutation selects an
implicit `latest`. Historical runs are discoverable and readable; control or
continuation requires archived source. Pause requests and completed safe-boundary
pauses are separate states. A live scheduler resume releases its pause; a stopped
driver resume verifies frozen source/environment/inputs and committed state.
Weights alone are not full continuation state.

Operation changes on stopped runs use `resume --profile ...` or explicit monitor
and resource flags and are appended to `execution.json`. Live scheduling uses
`batch-limit`, `batch-mode`, `batch-budget` and the existing terminal controls.
The config diff separates scientific data, operation settings and origins.
Results expose actual available metrics and paths; missing evidence stays missing.

## Monitor and sync

W&B defaults to online/project `SmartSOM`; configure entity or use the account
default. `--wandb disabled` keeps only local events; `--wandb offline` also keeps
uploadable SDK records. Missing enabled dependencies fail before launch. Online
initialization failure requires an explicit interactive offline choice; background
launches must have offline selected in advance. Mid-run sync failures do not
invalidate local checkpoints. Training and synchronization have separate status.
If `log()` rejects an event before the SDK accepts it, `upload_errors` and the last
failed event sequence remain in tracking status across polling, shutdown and
continuation. A later successful SDK queue flush cannot prove that event was
uploaded. The complete event remains in the local journal; SDK offline sync does
not automatically replay rejected journal events.

Local `tracking/events.jsonl`, manifest, frozen inputs, progress, cases and
continuation checkpoints are authoritative. W&B URLs in status and terminal
output can be opened on a phone. The logger does not upload source or checkpoints
by default. Online continuation pins run ID/project/entity. Each offline segment
has its own ID and sync directory, with the shared experiment ID in its config.
Use each recorded `sync_command` from `tracking/status.json` to upload its segment;
do not assume offline segments merge into one online history.

`--tensorboard` writes events only. Start a localhost viewer separately:

```sh
smartsom tensorboard RUN_ID --port 6006
```

Shell completion is opt-in; SmartSOM does not edit shell startup files:

```sh
# Bash
source <(smartsom completion bash)
# Zsh (after compinit)
source <(smartsom completion zsh)
```

Historical test/source retirement mappings are in `docs/archive/`; archived
physics is never rerun by the current historical viewer. Engineering verification
status is recorded separately in the migration document.

## Scientific matrices

An Experiment may declare explicit scientific override axes. Each combination
becomes one independent frozen experiment, scheduled under the same total
concurrency limit as all other files in the directory:

```yaml
matrix:
  experiment.run.seed: [101, 102, 103]
  algorithm.social_arm: [base, all]
```

Matrix axes use the same dotted scientific fields as advanced `--set`. Operational
axes are rejected. Explicit CLI science overrides apply after matrix values;
if they collapse distinct entries into duplicate scientific identities, compilation
rejects the batch. The matrix does not create a second nested worker pool.

Resource budgets are logical admission limits, not OS enforcement. Standalone
native runs check one CPU and observed/floor memory before execution; directory
and Tune schedulers additionally account for active workers and host headroom.
Tune performance calibration is explicit; adaptive runtime control preserves the
selected numerical-thread layout and changes independent experiment concurrency.
