# Create and inspect a scenario

A scenario combines a factory, a workload and enabled modules. A run adds an
algorithm and seed. Start from an existing template or import a traditional FJS
file; every generated project contains its own inputs and relative references,
so the whole directory can be moved to another location or machine.

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

Creation returns the project directory. It writes:

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
uv run --no-sync smartsom validate my-scenario/run.yaml
uv run --no-sync smartsom run my-scenario/run.yaml
```

Results go under `my-scenario/runs/`. For the learning template, install the
required extra in the environment used to run SmartSOM, then execute its
`train.yaml` with `smartsom train`. Its algorithm and 4096-round budget start from
the existing micro preset; creating or previewing the template does not train it.
See [usage](usage.md) for the learning dependency and execution commands.

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
uv run --no-sync smartsom run imported-example/run.yaml
```

FJS input supplies processing routes and times. It does not supply a transport
layout, buffer capacities or disturbances; add those through the existing
factory and scenario contracts when needed.
