# SmartSOM

SmartSOM is intended to become a modular, event-driven simulator and experiment
platform for dynamic flexible job shop scheduling problems. The long-term
direction includes composable dynamic events, production resources and
constraints, online policies, optimization solvers, evolutionary methods, and
optional single-agent and multi-agent learning adapters.

## Status

The first deterministic static core slice is implemented: separate immutable
factory/workload inputs, serial job chains, one or more processing modes per operation,
semantic dispatch, step/run/replay, stable completion ordering, in-memory trace,
actual makespan, and transition invariants.

The single-run configuration slice also supports strict five-file configuration,
seeded static JSP/FJSP generation or instance import, deterministic dispatch policies,
`validate`/`run`/`import-fjs` commands, and persisted run evidence including failures.

Static JSP also supports explicit waiting, exact schedule replay, SPT, and an
optional PyJobShop/CP-SAT adapter. The fixed ft06 reference and a real CP solution
both replay to makespan **55**. Static FJSP supports multiple modes, including
distinct modes on the same machine, and traditional `.fjs` import. The official
PyJobShop small example and Mk01 have fixed references and real adapter solutions
that replay to **6** and **40**. Online arrivals now add independent release/reveal
timing, filtered job observations, explicit event waiting and reproducible arrival
generation. Processing-time uncertainty adds independently materialized actual
durations while online policies continue to observe nominal durations.
Batch execution, other dynamic modules, and learning frameworks
remain planned. CP runtime classification is deferred.
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
`select_action(context) -> Dispatch | WaitUntil | WaitNextEvent`. Rejected actions raise `InvalidActionError`
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

Trace consumers can use `Simulator.trace_since(cursor)` to read an immutable
suffix without copying earlier records. The cursor is an integer from zero to
the current trace length; reading never advances or consumes the simulation.
The full `trace` and completed result remain available in memory.

Runs save their resolved inputs, reusable `realized_instance.json`, manifest,
trace, metrics, summary, and progress under the configured output root. Failures
retain available evidence and a `failure.json`; CLI errors return nonzero.
See [configured-run validation and file contracts](docs/validation/configured-runs.md)
for the supported fields, generation rules, and exact acceptance cases.

## Static JSP and FJSP algorithms

```bash
uv run smartsom run configs/runs/ft06_spt.yaml
uv sync --locked --extra cp
uv run --extra cp smartsom run configs/runs/ft06_cp.yaml
uv run --extra cp smartsom run configs/runs/pyjobshop_fjsp_cp.yaml  # 6
uv run --extra cp smartsom run configs/runs/mk01_cp.yaml           # 40
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

## FJSP inputs

```bash
uv run smartsom run configs/runs/fjsp_fast.yaml            # selected fast mode: 4
uv run smartsom run configs/runs/fjsp_slow.yaml            # selected slow mode: 7
uv run smartsom run configs/runs/generated_fjsp_spt.yaml
uv run smartsom import-fjs data/reference/mk01/Mk01.fjs \
  --instance-id mk01 --output-dir artifacts/imported-mk01
```

The import command validates the whole input before creating a new directory,
then writes `factory.yaml` and `workload.json` for existing scenario/run files.
It refuses an existing output directory. The Python API is
`smartsom.workloads.import_fjs(path, instance_id="mk01")` and returns an immutable
`ImportedProblem` with factory, workload and import provenance.

[`static_fjsp.yaml`](configs/workloads/static_fjsp.yaml) samples candidate machines
independently per operation, one mode per candidate, then independently samples
each mode's fixed nominal duration. Operations may revisit machines. Only the
run's derived workload seed is consumed; exporting and reimporting an instance
keeps its content fixed when the run seed changes. The JSP generator retains its
original sampling recipe and golden output.

Handwritten instances can retain several modes on one machine, even with equal
durations and different IDs. Dispatch selects one mode for the entire operation.
SPT chooses the shortest currently legal pair; it does not wait for a busy faster
machine. See the [item 4 acceptance record](docs/validation/static-fjsp.md) for
the import grammar, generator recipe, source snapshots and replay checks.

## Online job arrivals

```bash
uv run smartsom validate configs/runs/online_arrivals_event.yaml
uv run smartsom run configs/runs/online_arrivals_dispatch.yaml  # makespan 6
uv run smartsom run configs/runs/online_arrivals_event.yaml     # makespan 6
uv run smartsom run configs/runs/generated_arrivals_event.yaml
```

`scenario.arrivals` selects a fixed timing table or `uniform_release_v1` profile.
Reveal exposes a full job and its release time; release permits processing.
Before reveal the job is absent from all decision fields. The default
`dispatch_available` trigger auto-advances when no dispatch is legal;
`arrival_event` also exposes arrival notifications with empty candidates.
SPT/first-feasible then return `WaitNextEvent()` without learning the next event
clock. Scripted policies must explicitly provide that action.

The Python engine and both replay APIs accept `arrivals=ArrivalPlan(...)` and
`decision_trigger=...`. All-zero plans preserve previous static traces exactly.
Runs additionally save `realized_events.jsonl` for reuse and `observations.jsonl`
for the snapshots actually delivered to policies. Workload and arrival digests
remain separate. Dynamic scenarios reject the static CP provider. See the
[item 5 contract and acceptance](docs/validation/online-arrivals.md) for the
schema, generator recipe, information boundary and independent hand reference.

## Processing-time uncertainty

```bash
uv run smartsom validate configs/runs/processing_generated.yaml
uv run smartsom run configs/runs/processing_fixed.yaml       # makespan 20
uv run smartsom run configs/runs/processing_generated.yaml   # makespan 23
uv run smartsom run configs/runs/processing_arrivals_event.yaml
```

`scenario.processing_time` selects a fixed realized table or an independent
`uniform_multiplier` profile (default 0.8–1.2). Workloads retain nominal durations;
the engine executes the chosen mode's actual duration. Candidates and SPT use
nominal values, and only completion reveals the selected mode's actual duration.
Unselected modes' realizations stay private.

`ProcessingTimePlan` is accepted through the `processing_times` keyword on
`Simulator`, `replay` and `replay_schedule`. Versioned semantic-ID draws and exact
half-up rounding preserve deterministic replay without runtime sampling.
`realized_processing_times.json` can be imported directly under a different seed
or provider. Unit multipliers preserve prior complete traces. Processing-time
uncertainty can compose with arrivals; it rejects the static CP provider.
See the [item 6 acceptance record](docs/validation/processing-times.md).

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
