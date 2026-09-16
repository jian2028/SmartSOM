# Create and inspect a scenario

Studio edits the factory's grid, resources, capabilities and ports. A scenario
combines that factory with a workload and disturbance settings. A run adds the
algorithm, seed, budgets and output options. References are relative to the file
that declares them, so a complete project can be moved together.

## Run the default generated example

```sh
uv sync --locked
source .venv/bin/activate
smartsom validate --config configs/runs/run_test.yaml
smartsom show-config --config configs/runs/run_test.yaml
smartsom run --config configs/runs/run_test.yaml
```

The example has two machines, one AGV, explicit ports and finite PRE/POST pools
of capacity 8. SPT selects processing and transport decisions through the grid
core. Seed 42 generates two jobs with two steps each and durations from 1 to 5.
The checked example finishes at tick 30. Validation and preview do not advance
simulation or allocate a run directory.

Edit `configs/workloads/workload_test.yaml` to change the generated workload:

```yaml
schema: smartsom.workload/v2
profile:
  jobs: 2
  operation_types: [operation_1, operation_2]
  min_operations: 2
  max_operations: 2
  nominal_min: 1
  nominal_max: 5
  due_at: 10000
```

Steps select types from the catalog; repeated types are allowed. Alternatively,
`route: [operation_1, operation_2]` specifies a fixed route in place of
`operation_types`. Machine eligibility comes from the factory's capability
catalog. Increasing the job count does not increase buffer capacity.

`smartsom run --preset test` uses the packaged defaults. It does not load local
edits to repository files. The separate `configs/runs/production_hand.yaml`
contains the one-machine eight-tick example used for hand checks.

## Reuse a generated workload as fixed JSON

A normal run creates `run.json` and optional `trace.jsonl` in the printed run
directory. The manifest retains frozen inputs, original generation settings,
source identity and outcome. There is no separate realized-instance sidecar.

Export the frozen workload using the existing snapshot API:

```python
from pathlib import Path
from smartsom.config import load_resolved_run

prepared = load_resolved_run("runs/your-run/run.json")
Path("configs/workloads/workload_test.json").write_text(
    prepared.resolved.workload_json + "\n", encoding="utf-8"
)
```

Change the scenario reference to:

```yaml
workload: ../workloads/workload_test.json
```

The JSON document has `schema: smartsom.workload/v2` and explicit `demands`.
It is loaded directly even if the run seed changes. Workload freezing alone
does not freeze subsequently configured arrivals, outages or processing/quality
disturbances. To rerun the complete frozen experiment after removing its YAML
sources, pass `load_resolved_run(...)` to `smartsom.experiments.run_one`.
Offline playback instead reads the trace and does not rerun a policy.

To restore generation, change the reference back to `../workloads/workload_test.yaml`.
Both YAML and JSON can hold profiles or explicit demands; the contents establish
the contract, not the extension.

## Choose a starting point

| Template | Contents |
| --- | --- |
| `minimal_jsp` | An explicit two-machine grid with fixed workload steps |
| `generated_fjsp` | A grid factory and seeded flexible workload profile |
| `transport_buffers` | AGVs, ports and finite buffers; omitted PRE/POST facilities stay absent |
| `marl_micro` | Four jobs, nine steps, eight machines, four AGVs, arrivals, holding and quality facilities |

```python
from smartsom.config.authoring import create_template, list_templates, preview_scenario

print(list_templates())
project = create_template("minimal_jsp", "my-scenario")
print(preview_scenario(project))
```

Or use the CLI:

```sh
smartsom init generated_fjsp my-scenario
smartsom validate --config my-scenario/run.yaml
smartsom run --config my-scenario/run.yaml
```

Template creation refuses an existing destination. It writes `factory.yaml`,
`workload.yaml`, `scenario.yaml`, `algorithm.yaml` and `run.yaml`. The `marl_micro`
template also supplies `learning-algorithm.yaml` and `train.yaml`; creation and
preview never start training. The ordinary baseline requires no learner or solver.

Edit grid geometry, resources, capabilities and capacities in the factory. Edit
operation types, nominal times, optional `machine_nominal_ticks`, and demand
release/due times in the workload. Put disturbance parameters in the scenario.
Algorithm selection, seeds, training controls and output paths belong to the run
or experiment configuration. Studio continues to edit only the factory.

## Import a complete FJS project

FJS supplies processing routes and alternative machine times. It supplies no
grid layout or port positions. Author a compatible factory first:

```python
from smartsom.config.authoring import import_fjs_project, preview_scenario

project = import_fjs_project(
    "inputs/example.fjs", "imported-example", factory="my-factory.yaml"
)
print(preview_scenario(project))
```

Machine labels `M1`, `M2`, and so on must exist in the factory, or be explicitly
mapped through `machine_map`. Each imported operation needs a catalog type whose
capable machine set exactly matches its alternatives. The importer preserves
per-machine durations as `machine_nominal_ticks`; these overrides never add
capability. Repeated alternatives for the same machine are rejected by grid
import because they cannot represent distinct quality modes implicitly.

The project retains the source bytes in `source.fjs` and import provenance in
`workload.yaml`, alongside the factory, scenario, algorithm and run files. The
source filename stem is the default instance ID. Invalid input fails before
creating the destination; imported projects can be moved and executed after the
original source is removed.

Historical matrix inputs remain tied to their original source checkout. The
current runner requires explicit grid geometry and never guesses it from a
travel-time matrix. See [runtime contracts](production-runtime.md) and
[experiment workflows](experiments.md) for training, evaluation and recording.
