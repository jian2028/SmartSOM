# Current scientific inputs

Scenario, Algorithm and Experiment remain the daily authoring inputs. Current
Scenario documents intentionally embed geometry together with their job and fault
rules; they are not additional standalone Factory files. The eight selected
Small/Medium × fault scenarios use that same contract.

The only four standalone factory resources are in
[`src/smartsom/config/maps/`](../src/smartsom/config/maps/):

| File | Grid | Machines / AGVs |
| --- | --- | --- |
| `small-factory.yaml` | 22 × 6 | 8 / 8 |
| `medium-factory.yaml` | 27 × 12 | 20 / 20 |
| `small-wide-factory.yaml` | 24 × 6 | 8 / 8 |
| `medium-wide-factory.yaml` | 29 × 19 | 20 / 20 |

Studio and installed packages load those exact resources. No duplicate factory
presets live in this directory. Expanded Factory/Workload inputs remain supported
for existing callers and frozen execution inputs. The map catalog does not select
fault conditions, training budgets, or experiment results.

Existing workloads, algorithms, experiment definitions and scenarios are retained.
The historical `test/` inputs include retired schemas and references to removed
maps; they require their original Git source and are not current runnable recipes.
See [the retirement ledger](../docs/archive/four-map-retirement.json) for exact
paths, hashes and source recovery, and [scene authoring](../docs/scenario-authoring.md)
for current use. Old running worktrees and generated run artifacts are unchanged.
