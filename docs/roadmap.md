# SmartSOM Roadmap

Status: M0 repository foundation.

Each milestone is a bounded vertical slice. A later milestone does not begin
merely because its directories or interfaces have been discussed.

## Next implementation slices

The immediate Week 2 sequence narrows the broader milestones below:

1. Implement the static core subset of M1/M2: separate factory/workload inputs,
   serial job chains, one mode per operation, semantic step/run/replay, integer
   completion events, immutable decisions, trace, makespan, and invariants.
   Validate resource competition and crossing routes with two hand-computable
   fixtures. No config, CLI, dynamic module, solver, or framework is included.
2. Implement the five-file config/resolver/evidence slice through
   `run_one(ResolvedRun)`, preserving exact equivalence to code-built fixtures.
3. Add static JSP policies and a CP adapter. Define intentional waiting before
   claiming exact replay of external schedules with idle time.
4. Add and validate multiple processing modes and FJSP import/solver mapping.

The first slice does not complete the full FJSP domain or experiment milestones.
Their wider capabilities remain planned until separately implemented and tested.

## M0 — Repository Foundation

Establish an installable Python 3.12 package, locked `uv` environment, quality
gate, concise project governance, architecture contract, and roadmap. No
simulator behavior is included.

Exit criteria:

- the package installs and imports;
- Ruff and pytest pass locally and in CI;
- the README states that the repository is scaffold-only;
- no unimplemented future feature packages exist.

## M1 — Static Domain

Implement typed jobs, operations, machines, processing alternatives, scenario
validation, and one tiny static FJSP fixture. Keep the domain independent of
algorithm and I/O frameworks.

## M2 — Deterministic Static Engine

Implement an event calendar, operation completion, semantic dispatch, ready
sets, feasible action views, and deterministic transitions. Identical inputs,
seeds, and actions must produce identical schedules and records.

## M3 — Policies, Experiments, and Replay

Add random-valid and shortest-processing-time policies, then implement typed
single-run configuration, local artifacts, progress reporting, semantic trace,
metrics, replay, and the `smartsom run` CLI. Add batch composition only after
single-run behavior is stable. A small-instance CP solver remains optional.

## M4 — Dynamic FJSP

Add online job arrival before machine breakdown and repair. Require
no-look-ahead observations, deterministic same-time event ordering,
module-disabled equivalence, and dynamic semantic replay.

## M5 — Production Logistics and Energy

Add finite buffers before simple transport capacity. Add route-level AGV
conflicts, charging, and traffic only after basic transport semantics are
validated. Introduce energy first as a metric/objective and add hard energy
constraints only when a research question requires them.

## M6 — Learning

Add a Gymnasium single-agent adapter before a DRL learner. Add W&B only as an
optional telemetry sink. PettingZoo and MARL wait for stable staged or joint
decision semantics and do not alter the canonical simulator core.
