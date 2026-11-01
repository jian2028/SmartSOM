# Current runtime migration

The earlier runtime consolidation has recorded acceptance. The subsequent algorithm
component change has its own [validation record](../validation/algorithm-components.md) and final review/CI gate.
See [current commands](../current-workflow.md) and [engineering acceptance](../validation/runtime-consolidation.md).
See ADR0058 and the source inventory in ../archive/index.json.

## Current entry points

Use the existing four-file Base schemas. Schema versions identify serialized
inputs; they no longer select historical physics. The built-in observation layout
is native-v2. Historical checkpoints are rejected rather than migrated.

```sh
uv run smartsom check configs/base-diagnostic/small.yaml
uv run smartsom run configs/base-diagnostic/small.yaml --task smoke
uv run smartsom batch-run PATH_TO_EXPERIMENT_DIRECTORY --foreground
uv run smartsom resume PATH_TO_CURRENT_RUN
```

Training and evaluation require calibration evidence. Algorithm schema `smartsom.algorithm-components/v1` selects
explicit learner, observation and encoder factories, production role bindings,
action permissions and an independent Mover. See [component configuration and
migration](../algorithm-components.md) and ADR0059. The earlier flat Algorithm
schema is accepted only by `smartsom config migrate`, never by current run/check.
Centralized rule/random/custom control remains supported; centralized DRL does not.

## Tracking

Training/evaluation resolve tracking.mode to online by default and project to
SmartSOM. Explicit YAML settings override WANDB_MODE/WANDB_PROJECT/WANDB_ENTITY.
Compilation freezes the resolved settings without connecting to W&B.
Credentials are supplied outside YAML. Install the optional wandb extra to run
online or offline tracking; disabled does not require the SDK.

Interactive foreground initialization failure asks whether to continue offline.
Noninteractive startup fails unless offline was explicitly selected. Runtime
upload failures retain local checkpoints and metric events. Inspect
tracking/status.json and progress.json independently for training and sync status.
The saved URL can be opened in a phone browser.

Online continuation reuses its recorded run ID and project/entity. Offline
continuation creates a separate run segment with a common experiment identity.
tracking/status.json lists each segment and its wandb sync command. Upload each
offline segment separately; this does not promise a merged cloud history.

## Historical evidence and recovery

The original checkout contains backup/2026-10-28-runtime-consolidation.
The component-refactor worktree additionally contains
backup/2026-10-29-algorithm-components/source.tar.gz, including its pre-change
lockfile and dirty-map source manifest. This ignored archive is local; the PR
contains the index and checksum, not the archive payload.
The archive index records source commits and verified SHA256 checksums.
The tar files include source, tests, pyproject.toml and uv.lock.
Existing runs, artifacts and frozen-source directories remain in place.

Extract the matching archive into a separate directory and rebuild its locked
environment with uv. Match the original run's commit, dirty-source evidence,
extensions, observation contract and dependency metadata before loading anything.
Never overwrite or resume an old run using this branch.

A source snapshot alone does not prove resumability. Model-only packages may
permit inference but omit optimizer, replay, target-network or random state.
An older Batch05 component was rejected by the pre-consolidation main loader.
A representative profiling-handoff PPO component now loads successfully using its
matching preserved dirty source (indexed separately); it remains model-only.
Do not infer evaluation or complete continuation from a successful weight load.

## Acceptance state

See the current [engineering acceptance record](../validation/runtime-consolidation.md)
for independent review, Base trace comparison, native learning/control, SDK events,
archive evidence and bounded performance measurements. PR checks provide final
Linux/macOS gate status. Online W&B cloud connectivity and full historical training
continuation are not claimed. This change launches no formal research experiments.
