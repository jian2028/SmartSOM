# Create and inspect a scenario

A scenario combines a factory, a workload and enabled modules. A run adds an
algorithm and seed. Start from an existing template or import a traditional FJS
file; every generated project contains its own inputs and relative references,
so the whole directory can be moved to another location or machine.

## Run the default generated example

From the repository root, install and activate the environment once. Activate it
again in each new terminal; subsequent commands can use `smartsom` directly:

```sh
uv sync --locked
source .venv/bin/activate
smartsom validate --config configs/runs/run_test.yaml
smartsom show-config --config configs/runs/run_test.yaml
smartsom run --config configs/runs/run_test.yaml
```

`run_test.yaml` references `scenario_test.yaml` and `algorithm_test.yaml`.
The scenario references `factory_test.yaml` and `workload_test.yaml`, in their
respective `configs/` subdirectories. The algorithm is SPT. The workload YAML is
a generation profile, so increasing the number of jobs requires only one edit:

```yaml
schema: smartsom.workload-profile/v1
generator: static_jsp_v1
profile:
  order_count: 1
  jobs_per_order: 2
  operations_per_job: {min: 2, max: 2}
  nominal_ticks: {min: 1, max: 5}
```

The factory supplies two machines, and `run_test.yaml` sets seed 42. Keeping
parameters and seed fixed reproduces the same generated workload. For this JSP
generator, each job visits a machine at most once; the maximum operation count
must not exceed the machine count. Use the FJSP template below for flexible
machine alternatives.

`smartsom run --preset test` runs the packaged defaults without specifying a
file. It does not read edits to your repository configs; use `--config` for those.

## Reuse a generated workload as fixed JSON

Each simulation prints an experiment `run_dir`. Its `run.json` records the
relative evidence directory in `paths.evaluation`; inside that directory,
`realized_instance.json` contains every generated job and its generation
provenance. Copy that file to `configs/workloads/workload_test.json`, then replace
the workload reference in `configs/scenarios/scenario_test.yaml` with:

```yaml
workload:
  kind: instance
  path: ../workloads/workload_test.json
```

Run the same validation and execution commands again. The fixed JSON is loaded
directly; it is not regenerated, even when the run seed changes. Keep the same
factory, algorithm and seed to reproduce the default example's schedule. Other
enabled disturbances have separate inputs and seeds; freezing the workload alone
does not freeze those disturbances.

To return to generation, restore `kind: profile` and
`path: ../workloads/workload_test.yaml`. The resolver accepts YAML and JSON for
both profiles and instances; `kind` describes the content, not the file extension.
Fixed instances can also be handwritten or imported from FJS.

## Choose a starting point

| Template | Contents | Useful first edit |
| --- | --- | --- |
| `minimal_jsp` | Two machines, two jobs, four fixed operations | Edit nominal durations in `workload.json` |
| `generated_fjsp` | Three machines and a seeded flexible workload profile | Edit job counts, operation counts or eligible machines in `workload-profile.yaml` |
| `transport_buffers` | Two machines, two AGVs and finite buffers, including a zero-capacity prebuffer | Edit travel times or buffer capacities in `factory.yaml` |
| `marl_micro` | Existing four-job, nine-operation learning micro with eight machines, four AGVs, arrivals, shared holding and quality | Run the SPT baseline, then use the separate training configuration |

The templates reuse the repository's current examples. Their base inputs and
existing import or generation provenance are preserved. They are starter cases;
editing them does not change a frozen acceptance recipe or establish a research
result. Template creation refuses an existing destination, including an empty
directory.

From Python, after installing SmartSOM:

```python
from smartsom.config.authoring import create_template, list_templates, preview_scenario

print(list_templates())
project = create_template("minimal_jsp", "my-scenario")
print(preview_scenario(project))
```

For a generated project, start with:

```sh
smartsom init generated_fjsp my-scenario
smartsom validate --config my-scenario/run.yaml
smartsom run --config my-scenario/run.yaml
```

Creation returns the project directory. It writes role-named files:

```text
my-scenario/
  factory.yaml
  workload.json             # workload-profile.yaml for generated_fjsp
  scenario.yaml
  algorithm.yaml            # builtin.spt
  run.yaml
```

The `marl_micro` template additionally supplies `learning-algorithm.yaml` and
`train.yaml`. The normal `run.yaml` remains a ready-to-run SPT baseline and needs
no optional learning framework.

## Preview and edit

`preview_scenario()` accepts either the scenario YAML file or its project
directory. It uses the normal resolver to validate references, materialize the
chosen seed and report counts, enabled modules, effective seeds and input hashes.
It does not construct a simulator, run a policy, start training or create output
directories. Counts describe the authoring inputs, including unrevealed jobs;
they are not observations delivered to a learning agent.

```python
summary = preview_scenario("my-scenario/scenario.yaml", seed=101)
print(summary["counts"])
print(summary["modules"])
print(summary["input_sha256"])
```

Edit each value where it belongs:

- `factory.yaml`: machine and AGV resources, travel matrix and buffer capacities.
- `workload.json`: fixed jobs, precedence and alternative processing modes.
- `workload-profile.yaml`: generation ranges; preview with a fixed seed to inspect
  the same generated instance again.
- `scenario.yaml`: references and enabled arrivals, processing-time uncertainty,
  machine outages, transport, buffers, holding and quality.
- `run.yaml`: algorithm reference, seed and output location.

All references are relative to the file declaring them. For example, a scenario
in `settings/scenario.yaml` can refer to `../factory.yaml`; changing the shell's
working directory does not change that reference. Move the complete project
directory together. Structural changes still need to satisfy the existing
domain and module contracts, which the preview checks through the resolver.

Execute the generated baseline with the existing CLI:

```sh
smartsom validate --config my-scenario/run.yaml
smartsom run --config my-scenario/run.yaml
```

Results go under `my-scenario/runs/`. For the learning template, install the
required extra in the environment used to run SmartSOM, then execute its
`train.yaml` with `smartsom train`. Its algorithm and 4096-round budget start from
the existing micro preset; creating or previewing the template does not train it.
See [the experiment guide](experiments.md) for learning dependencies and commands.

## Import a complete FJS project

```python
from smartsom.config.authoring import import_fjs_project, preview_scenario

project = import_fjs_project("inputs/example.fjs", "imported-example")
print(preview_scenario(project))
```

The importer uses the existing strict traditional serial-FJSP parser: one job per
nonempty line, one-based machine numbers, positive integer durations and all
declared alternatives preserved. The source filename stem becomes the instance
ID. Invalid input fails before the destination directory is created.

The new project includes `factory.yaml`, `workload.json`, `scenario.yaml`,
`algorithm.yaml`, `run.yaml` and an exact `source.fjs` copy. `workload.json` retains
the parser version, source-byte digest, instance ID and header. It can run and
preview after the original source has been moved or removed:

```sh
smartsom run --config imported-example/run.yaml
```

FJS input supplies processing routes and times. It does not supply a transport
layout, buffer capacities or disturbances; add those through the existing
factory and scenario contracts when needed.
