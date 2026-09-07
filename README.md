# SmartSOM

SmartSOM is intended to become a modular, event-driven simulator and experiment
platform for dynamic flexible job shop scheduling problems. The long-term
direction includes composable dynamic events, production resources and
constraints, online policies, optimization solvers, evolutionary methods, and
optional single-agent and multi-agent learning adapters.

## Status

The first deterministic static core slice is implemented: separate immutable
factory/workload inputs, serial job chains, one processing mode per operation,
semantic dispatch, step/run/replay, stable completion ordering, in-memory trace,
actual makespan, and transition invariants.

The single-run configuration slice also supports strict five-file configuration,
seeded static JSP generation or instance import, deterministic dispatch policies,
`validate`/`run` commands, and persisted run evidence including failures.

Static JSP also supports explicit waiting, exact schedule replay, SPT, and an
optional PyJobShop/CP-SAT adapter. The fixed ft06 reference and a real CP solution
both replay to makespan **55**. This is a static JSP subset of the planned FJSP
domain. Multiple modes, batch execution, dynamic modules, and learning frameworks
remain planned.
The [static core validation record](docs/validation/static-core.md) gives the
exact hand-calculated cases, boundaries, and verification commands.

## Python API

```python
from smartsom.dispatch import Dispatch
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
)
from smartsom.engine import Simulator, replay

factory = FactorySpec((Machine("M1"),))
operation = Operation("operation", (ProcessingMode("standard", "M1", 2),))
job = Job("job", (operation,))
workload = WorkloadInstance((Order("order", (job,)),))
simulator = Simulator(factory, workload)
decision = simulator.current_decision  # Immutable; reading does not advance time.
result = simulator.step(Dispatch("operation", "standard"))
assert result.makespan == 2
assert replay(factory, workload, result.actions) == result
```

`step()` returns the next `DecisionContext` or a terminal `SimulationResult`.
`run(policy)` repeatedly calls that same method; a policy needs only
`select_action(context) -> Dispatch | WaitUntil`. Rejected actions raise `InvalidActionError`
before changing state. A new simulator starts a new episode; completed instances
cannot be advanced again.

`WaitUntil(tick)` intentionally advances time, returning early at a completion
that permits a decision. It creates no lasting wait commitment. Complete
`ScheduledOperation` intervals can be passed to `replay_schedule(factory,
workload, schedule)`; it validates the whole schedule and checks every actual
interval after execution, including intentional idle time.

## Configured Runs

```bash
uv sync --locked
uv run smartsom validate configs/runs/generated.yaml
uv run smartsom run configs/runs/competition.yaml  # makespan 6
uv run smartsom run configs/runs/crossing.yaml     # makespan 5
uv run smartsom run configs/runs/generated.yaml   # seeded static JSP
```

Each run file references a scenario and algorithm; the scenario references a
factory and exactly one workload profile or instance. Relative paths belong to
the containing file, so the command works from another directory with an
absolute run-config path. Root seeds belong only in the run file.

```python
from smartsom.config import resolve_run
from smartsom.experiments import run_one

resolved = resolve_run("configs/runs/generated.yaml")  # No output directory.
result = run_one(resolved)  # Uses the resolved snapshot without rereading files.
print(result.simulation_result.makespan, result.run_dir)
```

Runs save their resolved inputs, reusable `realized_instance.json`, manifest,
trace, metrics, summary, and progress under the configured output root. Failures
retain available evidence and a `failure.json`; CLI errors return nonzero.
See [configured-run validation and file contracts](docs/validation/configured-runs.md)
for the supported fields, generation rules, and exact acceptance cases.

## Static JSP algorithms

```bash
uv run smartsom run configs/runs/ft06_spt.yaml
uv sync --locked --extra cp
uv run --extra cp smartsom run configs/runs/ft06_cp.yaml
```

`builtin.spt` chooses the shortest current legal duration, breaking ties by
operation and mode IDs. It does not use future information or promise optimality.
`pyjobshop.cp_sat` requires explicit `full_static` visibility and uses the pinned
optional dependencies PyJobShop 0.0.9 and OR-Tools 9.12.4544, with one worker.
The run-owned `budget.solver_time_limit_seconds` defaults to 60 for CP and is
rejected for online providers. A missing extra produces an explicit error.

CP saves `solver_result.json` before replay. A complete feasible solution can
succeed without proof of optimality; the summary records `solver_status` and
`proven_optimal`. See the [item 3 acceptance record](docs/validation/static-jsp.md)
for the reference source snapshots, Python replay example, and base/CP checks.

## Design Direction

- Keep a thin semantic simulation core and add behavior through composition.
- Represent decisions with stable entity-based actions rather than transient
  candidate positions.
- Keep simulator truth independent of algorithm and learning frameworks.
- Use one execution path for manual runs and batch experiments.
- Preserve local manifests, progress logs, semantic traces, metrics, summaries,
  and failures as the authoritative experiment record.
- Add optional dependencies only with the adapter that needs them.

The planned boundaries are documented in
[`docs/architecture.md`](docs/architecture.md). The staged implementation order
is documented in [`docs/roadmap.md`](docs/roadmap.md).

## Development

SmartSOM uses Python 3.12 and `uv`.

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
# Required when changing the CP adapter:
uv sync --locked --extra cp
SMARTSOM_REQUIRE_CP=1 uv run --no-sync pytest -q
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before creating a branch or commit.
