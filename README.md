# SmartSOM

SmartSOM is a modular, event-driven simulator for dynamic flexible job shop
scheduling research. It combines production and material transport with
configurable disturbances, scheduling baselines and reinforcement learning.
Runs retain their inputs, decisions and schedules for reproducible evaluation
and exact replay.

## Features

- **Scheduling:** static JSP/FJSP, multiple processing modes, online arrivals,
  processing-time uncertainty, and machine breakdowns and repairs.
- **Production logistics:** AGV transport, finite buffers, reservations, blocking,
  shared holding buffers, and quality–speed processing modes.
- **Algorithms:** scripted and dispatching policies, optional CP-SAT for static
  cases, centralized RLlib PPO / SB3 MaskablePPO, and resource-agent MARL with
  PettingZoo and separate shared machine / AGV PPO policies.
- **Experiments:** YAML configuration, seeded paired studies, bounded batch
  execution, checkpoint evaluation, and action / schedule / joint-decision replay.

The centralized and resource-agent learning paths have local macOS validation.
Linux/h20 validation is pending. Current learning checkpoints support the same
base structure with different random realizations; training resume and
cross-structure generalization are not implemented. See the
[roadmap](docs/roadmap.md) and [validation records](docs/validation/) for scope
and evidence. Short training runs do not establish an advantage over baselines.

## Quick start

Requires **Python 3.12** and **uv**. From the repository root:

```sh
uv sync --locked
uv run --no-sync smartsom validate configs/runs/competition.yaml
uv run --no-sync smartsom run configs/runs/competition.yaml
```

The included example has a makespan of **6**. Its output directory contains the
resolved configuration, manifest, trace and summary. Output locations are defined
by the run configuration; this example writes under `runs/`.

A scenario selects the factory, workload and enabled modules. An algorithm file
selects the policy. A run file combines them with the seed, budget and output
settings. Browse [configs/](configs/) for examples.

To preview or execute a paired study:

```sh
uv run --no-sync smartsom plan configs/studies/quality_compare.yaml
uv run --no-sync smartsom batch configs/studies/quality_compare.yaml --workers 2
```

## Optional solvers and learning

Install only the extras you need. For a static CP-SAT example:

```sh
uv sync --locked --extra cp
uv run --no-sync smartsom run configs/runs/ft06_cp.yaml
```

For centralized PPO and resource-agent MARL:

```sh
uv sync --locked --extra learning-rllib --extra learning-marl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Centralized PPO
uv run --no-sync smartsom train configs/runs/learning_rllib.yaml

# Resource-agent MARL: shared machine and AGV policies
uv run --no-sync smartsom train configs/runs/learning_marl.yaml
```

SB3 MaskablePPO uses the `learning-sb3` extra and
`configs/runs/learning_sb3.yaml`. Interface-only extras are `gym` and `pettingzoo`.

Training produces a `checkpoint_algorithm.json` file. Reference it as an
algorithm in a run or study configuration to evaluate the saved model; `run` and
`batch` do not start training. Detailed commands and acceptance procedures are in
the [usage guide](docs/usage.md).

## Repository structure

```text
src/smartsom/     Simulator, domain models, configuration, algorithms and learning
configs/         Factory, workload, scenario, algorithm, run and study examples
data/reference/  Reference inputs and small validation cases
scripts/         Input preparation and acceptance commands
tests/           Unit and integration tests
docs/            Usage, architecture, decisions, validation and paper plans
runs/            Local run output (not committed)
artifacts/       Local training, evaluation and replay evidence (not committed)
```

## Documentation and development

- [Usage guide](docs/usage.md) — detailed CLI, Python API and module examples.
- [Architecture](docs/architecture.md) and [decisions](docs/decisions/) — contracts
  and design rationale.
- [Roadmap](docs/roadmap.md) and [validation](docs/validation/) — implemented scope
  and engineering acceptance.
- [Paper plans](docs/papers/README.md) — research outlines and reproduction plans.
- [Contributing](CONTRIBUTING.md) — environment, tests and commit conventions.

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Optional solver and learning acceptance tests require their corresponding extras;
see the validation records for the required checks.
