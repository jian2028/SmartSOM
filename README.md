# SmartSOM

> Grid production, training and playback share one runtime.
> See [current commands and verification status](docs/production-runtime.md).
> The original experiment workflows now use the single grid core. Native UI
> interaction checks remain incomplete; historical acceptance results remain
> tied to their original source commits.

SmartSOM is an event-driven simulator for dynamic flexible job shop scheduling
research. It models production, AGV transport, finite buffers and disturbances,
with scheduling rules and three PPO learning backends. Experiments retain their
inputs, decisions and committed states for audit and recorded playback. CP-SAT
does not yet support the grid physics and is explicitly rejected.

## Quick start

Use **Python 3.12** and **uv**. From the repository root, install the base
environment and activate it:

```sh
uv sync --locked
source .venv/bin/activate

smartsom validate --config configs/runs/run_test.yaml
smartsom show-config --config configs/runs/run_test.yaml
smartsom run --config configs/runs/run_test.yaml
```

Activate the environment again in each new terminal. All commands below assume
it is active. Alternatively, prefix a command with `uv run --no-sync` to use the
project environment without activation.

This first simulation uses two machines and SPT scheduling. It generates two
jobs with two operations each and processing times from 1 to 5, using run seed
42. It needs no learning framework or solver. `validate` and `show-config`
resolve the inputs without advancing the simulator; `run` executes the policy.

The five input files have separate responsibilities:

| File | What to configure |
| --- | --- |
| [factory_test.yaml](configs/factories/factory_test.yaml) | Grid positions, machine capabilities, one AGV, ports and explicit finite PRE/POST buffers. |
| [workload_test.yaml](configs/workloads/workload_test.yaml) | Generation parameters: jobs, operation types, route length and duration ranges. |
| [scenario_test.yaml](configs/scenarios/scenario_test.yaml) | Factory and workload references; arrival, processing and outage settings when needed. |
| [algorithm_test.yaml](configs/algorithms/algorithm_test.yaml) | Scheduling policy, here `builtin.spt`. |
| [run_test.yaml](configs/runs/run_test.yaml) | Scenario and algorithm references, seed, objective and output directory. |

```text
run_test.yaml
├── scenario_test.yaml
│   ├── factory_test.yaml
│   └── workload_test.yaml
└── algorithm_test.yaml
```

Change `profile.jobs` in `workload_test.yaml` to generate more jobs;
there is no need to write every job by hand. References are relative to the YAML
file declaring them. Each generated step selects an operation type from the configured catalog; repeated
types are allowed. Machine eligibility follows capabilities. The example explicitly
sets each PRE/POST pool to capacity 8; increasing the job count does not expand storage.

The command prints `makespan` and `run_dir`; the included parameters and seed
produce makespan **30** in the grid model. Each run directory under `runs/` contains
`run.json` (frozen inputs, source and result) and optional `trace.jsonl` (committed
states, actions and events). The separate `production_hand` example is the
one-machine, eight-tick hand-check case. Fixed workload JSON is also supported;
see [scenario authoring](docs/scenario-quickstart.md#reuse-a-generated-workload-as-fixed-json)
to reuse a generated instance.

For the bundled defaults, the shorter equivalent is:

```sh
smartsom run --preset test
```

The preset uses packaged inputs. Use `--config` to run your edits to the repository
files. The former `competition` preset and example paths have been removed; use
`test` or `configs/runs/run_test.yaml` instead.

A larger four-machine, four-AGV example is available in the
[Template 1 replay guide](docs/template1-replay.md), including static and
controlled-disturbance cases and independent validation commands.

## Train and evaluate

After the first simulation, optionally install all three learning backends, CPU
support, reports and local learning curves. `--inexact` preserves other extras
already installed in this environment:

```sh
uv sync --locked --extra learning --extra cpu --extra reports --extra tensorboard --inexact
smartsom doctor --preset marl_micro
smartsom show-config --preset marl_micro
smartsom train-evaluate --preset marl_micro --name first_marl
```

The default trains resource-agent MARL on the included micro case: seed 101,
4096 joint rounds and 16 PPO updates. Machine agents share one policy; AGV
agents share another. Independent evaluation uses seed 202 and five inputs.
This is an engineering example, not a claim of superiority over dispatching rules.

Commands print the experiment directory. Use that directory for the next steps:

```sh
smartsom evaluate RUN_DIRECTORY --checkpoint last --baseline spt
smartsom report RUN_DIRECTORY
smartsom audit RUN_DIRECTORY --training
smartsom export RUN_DIRECTORY --kind model
```

## Configure an experiment

Choose `marl_micro`, `rllib_micro` or `sb3_micro`. Previewing resolves configuration
and inputs without training or advancing the simulator.

```sh
smartsom presets list
smartsom show-config --preset marl_micro --steps 4096 --num-envs 4
smartsom train --preset sb3_micro --seed 101 --steps 4096 \
  --set algorithm.learning_rate=0.0003 --progress auto --verbose 1
```

Python uses the same implementation and typed configuration:

```python
from smartsom.api import load_preset, train

config = load_preset("marl_micro")
config.training.total_steps = 4096
config.algorithm.learning_rate = 3e-4
result = train(config)
print(result.run_dir, result.last_checkpoint)
```

Short scripts live in [examples/quickstart](examples/quickstart). See the
[experiment guide](docs/experiments.md) for configuration precedence, training,
validation, evaluation, resume, initialization and logging.

## New scenarios and optional dependencies

Create a portable generated workload project, then edit its YAML parameters:

```sh
smartsom init generated_fjsp my_scenario
smartsom validate --config my_scenario/run.yaml
smartsom run --config my_scenario/run.yaml
```

The generated project uses role-named files such as `factory.yaml` and
`workload-profile.yaml`. Available templates also include fixed JSP,
transport/buffers and MARL. FJS import creates a runnable project;
see [scenario authoring](docs/scenario-quickstart.md).

Install a single backend with `--extra learning-marl`, `--extra learning-rllib`
or `--extra learning-sb3`, together with `--extra cpu`. The micro presets enable
TensorBoard, so also include `--extra tensorboard`, or disable it with
`--set logging.tensorboard=false` (Python: `config.logging.tensorboard = False`).
The `cp` extra supports historical external references, not current grid execution.
TensorBoard, reports and W&B have separate extras;
W&B is off by default.
CPU and CUDA dependency profiles are mutually exclusive. Linux/CUDA execution
remains pending. Historical macOS acceptance at `4b4a7c2` describes that source,
not the current grid implementation; current verification is tracked separately.

## Factory design editor

SmartSOM Studio provides a local static grid editor with resource placement,
typed properties, a factory operation-type catalog and machine capability defaults,
slots/port bindings, undo, YAML saving, recovery and PNG/SVG
export. Documents start in Browse; click Edit to make changes. The compact four-machine
Template 1 is selected by default. Install the optional Qt
dependency and launch it with:

```sh
uv sync --locked --extra studio --inexact
smartsom studio
```

Studio and execution use the same `smartsom.factory/v2` design format. Studio
edits factories; evaluation and recorded playback use an independent window that
shares its drawing components. See [runtime and playback](docs/production-runtime.md),
[Studio](docs/studio.md) and [factory design](docs/factory-design.md).

## Structure and evidence

```text
src/smartsom/        Simulator, typed configuration, public API and learning
examples/quickstart/ Short train, evaluate and combined Python scripts
configs/            Default test inputs, recipes, factories, scenarios and paired studies
data/reference/     Reference instances and small validation cases
tests/              Unit and real backend integration checks
scripts/            Input preparation and frozen acceptance procedures
docs/               Usage, architecture, decisions and validation records
runs/               Authoritative local experiments; not committed
artifacts/          Historical and development evidence; not committed
```

Current implementation progress is tracked in [the refactor record](docs/implementation-usability.md).
Historical item 12/13 recipes and macOS evidence remain unchanged. The separate
[macOS acceptance record](docs/validation/usability-macos-4b4a7c2.md) records fresh
fixed-budget training, 25 completed evaluations and 164 passing feature checks.

- [Reports and portable exports](docs/reporting.md)
- [Architecture](docs/architecture.md), [decisions](docs/decisions/), [roadmap](docs/roadmap.md)
- [Validation records](docs/validation/) and [historical usage](docs/usage.md)
- [Contributing](CONTRIBUTING.md)

```sh
ruff check .
ruff format --check .
pytest -q
```

Optional integration checks require their corresponding locked extras. Passing
engineering checks does not establish a learning-performance conclusion.
