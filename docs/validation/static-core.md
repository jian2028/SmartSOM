# Static Core Validation

Implementation: static serial job chains with exactly one processing mode per
operation. Evidence: internal exact hand calculations and engineering tests.
These results do not claim an external benchmark reproduction or completion of
the full FJSP, experiment, or dynamic milestones.

## Supported boundary

- `FactorySpec(machines)` and `WorkloadInstance(orders)` are independent immutable
  domain owners. Caller-provided lists are copied into tuples.
- Jobs contain one complete serial chain defined by predecessor IDs, independent
  of collection order. Cross-job dependencies, branching, merging, empty inputs,
  unknown machines, and multiple modes are rejected.
- All jobs are initially available. Machines have unit capacity; processing is
  non-preemptive and lasts a positive integer number of ticks.
- `Dispatch(operation_id, processing_mode_id)` uses a workload-wide operation ID
  and an operation-local mode ID. The examples intentionally reuse `standard`
  across operations.
- The published feasible action view is authoritative. Rejected actions change
  neither state, clock, event queue, nor canonical trace.
- Policies receive immutable operation/machine snapshots and candidate
  machine/duration details. They do not receive mutable engine storage.
- With no legal dispatch, time advances directly to the next completion. Every
  completion at that tick is processed before a new decision. The completion
  calendar orders by `(tick, operation_id, processing_mode_id, machine_id)`.
- Trace records are immutable typed decision, dispatch/start, completion, and
  termination records. Actual completed intervals determine makespan; schedules
  are returned in `(start_time, operation_id)` order.
- `run(policy)` and semantic replay invoke the same public `step()` transition.
  A policy may continue a partially stepped episode. Completed instances reject
  further steps or policy runs; a fresh simulator starts a fresh episode.
- Invalid action, deadlock, invariant violation, and replay-length failures are
  explicit exceptions. They are not successful terminal results. Invariant
  failures indicate an engine defect; retrying that failed instance is not a
  recovery contract. Persisted failure reporting belongs to the later runner.

## Case 1: resource competition and simultaneous completions

Two jobs share the route `M1 -> M2`. A has durations `3, 2`; B has durations
`1, 3`. The scripted semantic action order is `B1, A1, B2, A2`.

| Operation | Machine | Start | Completion |
| --- | --- | ---: | ---: |
| B1 | M1 | 0 | 1 |
| A1 | M1 | 1 | 4 |
| B2 | M2 | 1 | 4 |
| A2 | M2 | 4 | 6 |

Hand-calculated and tested makespan: **6 ticks**.

| Tick | Canonical record order |
| ---: | --- |
| 0 | decision `{A1, B1}`; dispatch B1 |
| 1 | complete B1; decision `{A1, B2}`; dispatch A1; decision `{B2}`; dispatch B2 |
| 4 | complete A1; complete B2; decision `{A2}`; dispatch A2 |
| 6 | complete A2; terminate |

The test compares all 13 records, including sequence numbers, IDs, feasible
actions, durations, and the terminal reason. There is no decision between the
two completions at tick 4.

## Case 2: crossing routes and automatic time advancement

C follows `M1/2 -> M2/1`; D follows `M2/3 -> M1/2`. The action order is
`C1, D1, C2, D2`.

| Operation | Machine | Start | Completion |
| --- | --- | ---: | ---: |
| C1 | M1 | 0 | 2 |
| D1 | M2 | 0 | 3 |
| C2 | M2 | 3 | 4 |
| D2 | M1 | 3 | 5 |

Hand-calculated and tested makespan: **5 ticks**. C1 completes at tick 2, but
C2's machine remains busy. The engine records the completion, skips a decision
at tick 2, and exposes both successors only after D1 completes at tick 3.

## Verification

Fresh verification on 2026-09-07: **89 focused tests passed; 90 tests passed in
the full suite**. Locked offline environment synchronization, Ruff, format
checking, and whitespace checks passed. The README Python example was executed
successfully and local Markdown links were checked. No project dependency was
added or changed.

Focused checks:

```bash
uv run --locked pytest -q tests/unit
```

Repository quality gate:

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
git diff --check
```

In addition to both hand cases, tests cover:

- every legal action ordering of both tiny inputs, with an independent interval
  audit of counts, durations, precedence, resource exclusion, and makespan;
- exact step/run/repeat/replay equality and explicit short/long replay rejection;
- different simple test policies using the same candidate interface;
- original-container mutation, immutable snapshots/results, and pure getters;
- reordered machines/orders/jobs/operations and `PYTHONHASHSEED=1,17,321`;
- completion ordering under different dispatch and event insertion orders;
- one-machine consecutive operations, invalid actions before/after starts, and
  repeated calls after termination;
- injected loss of a completion event and an empty feasibility projection to
  verify invariant and deadlock diagnostics;
- imports that actively reject optional solver, learning, tracking, Pydantic,
  and YAML dependencies.

## Limitations and next slice

There is no explicit wait action. A legal external schedule containing
intentional idle time is outside this replay contract; add and validate that
behavior before the static JSP/CP adapter claims exact schedule reproduction.
There are no dynamic event types, module switches, configuration loader, CLI,
file-based trace writer, algorithm registry, or production algorithm provider.
The tests use local policy objects, not installed algorithm backends.

The next slice resolves the five-file authoring bundle into `ResolvedRun` and
connects that runner to this domain/step boundary. It must reproduce the same
canonical trace and metrics as code-constructed fixtures. Paths, manifests, and
provenance are separate from canonical simulator equivalence.
