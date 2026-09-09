# SmartSOM

SmartSOM is an event-driven simulator for dynamic flexible job shop scheduling
research. It models production, AGV transport, finite buffers and disturbances,
with scheduling rules, static CP-SAT and three PPO learning backends. Experiments
retain their inputs, decisions and schedules for audit and exact replay.

## Quick start

Use **Python 3.12** and **uv**. Install all three learning backends, CPU support,
reports and local learning curves:

```sh
uv sync --locked --extra learning --extra cpu --extra reports --extra tensorboard
uv run --no-sync smartsom doctor --preset marl_micro
uv run --no-sync smartsom show-config --preset marl_micro
uv run --no-sync smartsom train-evaluate --preset marl_micro --name first_marl
```

The default trains resource-agent MARL on the included micro case: seed 101,
4096 joint rounds and 16 PPO updates. Machine agents share one policy; AGV
agents share another. Independent evaluation uses seed 202 and five inputs.
This is an engineering example, not a claim of superiority over dispatching rules.

Commands print the experiment directory. Use that directory for the next steps:

```sh
uv run --no-sync smartsom evaluate RUN_DIRECTORY --checkpoint last --baseline spt
uv run --no-sync smartsom report RUN_DIRECTORY
uv run --no-sync smartsom audit RUN_DIRECTORY --training
uv run --no-sync smartsom export RUN_DIRECTORY --kind model
```

You can also activate `.venv` with `source .venv/bin/activate` and use `smartsom`
directly. `uv run --no-sync` does not require activation.

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

## Smaller installations and new scenarios

The simulator itself needs no Torch, Ray, Gymnasium or solver:

```sh
uv sync --locked
uv run --no-sync smartsom run --preset competition
uv run --no-sync smartsom init minimal_jsp my_scenario
uv run --no-sync smartsom run --config my_scenario/run.yaml
```

The competition example has makespan **6**. Available templates include JSP,
generated FJSP, transport/buffers and MARL. FJS import creates a runnable project;
see [scenario authoring](docs/scenario-quickstart.md).

Install a single backend with `--extra learning-marl`, `--extra learning-rllib`
or `--extra learning-sb3`, together with `--extra cpu`. The micro presets enable
TensorBoard, so also include `--extra tensorboard`, or disable it with
`--set logging.tensorboard=false` (Python: `config.logging.tensorboard = False`).
Add `--extra cp` for CP-SAT. TensorBoard, reports and W&B have separate extras;
W&B is off by default.
CPU and CUDA dependency profiles are mutually exclusive. Linux/CUDA execution
remains pending. The refactor passed fixed-source macOS CPU automated acceptance
at `4b4a7c2`; real-browser visual and download checks remain unverified.

## Structure and evidence

```text
src/smartsom/        Simulator, typed configuration, public API and learning
examples/quickstart/ Short train, evaluate and combined Python scripts
configs/            Historical recipes, factories, scenarios and paired studies
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
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Optional integration checks require their corresponding locked extras. Passing
engineering checks does not establish a learning-performance conclusion.
