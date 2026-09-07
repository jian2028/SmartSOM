# Static JSP solving and exact replay — Week 2 item 3

This slice adds intentional waiting, complete schedule replay, SPT and the
optional PyJobShop/CP-SAT provider to the existing static serial, single-mode
domain. The engine remains independent of configuration, storage and solver
packages. This is engineering acceptance on ft06 and hand-computable cases;
it does not establish FJSP support or general algorithm performance.

## Reproduce

From the repository root, the base environment supports SPT and fixed reference
replay without installing either PyJobShop or OR-Tools:

```bash
uv sync --locked
uv pip check
uv run --no-sync pytest -q
uv run smartsom validate configs/runs/ft06_spt.yaml
uv run smartsom run configs/runs/ft06_spt.yaml
```

The following reference replay uses only base dependencies and local fixtures:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from smartsom.config import resolve_run
from smartsom.domain import ScheduledOperation
from smartsom.engine import replay_schedule

resolved = resolve_run("configs/runs/ft06_spt.yaml")
reference = json.loads(Path("data/reference/ft06/schedule.json").read_text())
schedule = tuple(ScheduledOperation(**entry) for entry in reference["schedule"])
result = replay_schedule(resolved.factory, resolved.workload, schedule)
assert result.schedule == schedule
assert result.makespan == reference["makespan"] == 55
print(result.makespan)
PY
```

The CP environment must run the actual solver acceptance, with no skip accepted:

```bash
uv sync --locked --extra cp
uv pip check
SMARTSOM_REQUIRE_CP=1 uv run --no-sync pytest -q
uv run --extra cp ruff check .
uv run --extra cp ruff format --check .
git diff --check
uv run --extra cp smartsom run configs/runs/ft06_cp.yaml
```

`uv run --extra cp` retains the extra during environment synchronization;
`uv run --no-sync` uses the environment just synchronized. CI has separate base
and CP jobs. `SMARTSOM_REQUIRE_CP=1` makes missing CP dependencies a test failure.

## Action and schedule contracts

`SemanticAction = Dispatch | WaitUntil`, with both actions immutable.
`WaitUntil(until)` requires a strict integer greater than the current tick;
bools, floats and past/current ticks fail before mutation. Waiting records the
requested target and advances to that target or an earlier completion. Every
same-tick completion is processed in stable semantic order before a decision.
If no dispatch is legal, normal automatic advancement applies. A returned
decision always has at least one dispatch candidate.

There is no wait commitment after returning: the caller may dispatch immediately
or submit another wait. Waiting is legal when there are no future events, so an
initial operation of duration 2 can be delayed to `[5,7)`. For crossing routes
`C: M1/2 -> M2/1`, `D: M2/3 -> M1/2`, starting C1 at 0 then requesting a wait to
10 returns at C1's completion at 2. Starting D1 immediately gives D1 `[2,5)`,
C2 `[5,6)` and D2 `[5,7)`; the abandoned wait to 10 has no effect.

`Simulator.step()` and `OnlinePolicy.select_action()` now accept/return semantic
actions. `replay(factory, workload, actions)` records and repeats actual waits.
Unchanged dispatch-only cases retain the entire old trace, including sequence
numbers, as well as makespans 6 and 5.

`replay_schedule(factory, workload, schedule)` accepts an iterable of immutable
`ScheduledOperation(operation_id, processing_mode_id, machine_id, start_time,
completion_time)`. It validates complete coverage, IDs, strict nonnegative integer
times, exact durations, precedence and machine exclusion before constructing a
simulator. `ScheduleReplayPolicy` orders simultaneous starts by operation ID,
waits until the next requested start, and rejects any missed start. Its final
check compares every interval and recomputed makespan. The standalone API and
runner share this policy and check; every transition still uses `step()`.

## Configuration and solver boundary

The existing v1 envelopes gain these explicit capabilities:

| Owner | Fields and behavior |
| --- | --- |
| Scenario | `visibility: full_static` permits complete static input; `decision_context` remains the default. |
| SPT algorithm | `provider: builtin.spt`, empty parameters; online interface/current information defaults. |
| CP algorithm | `provider: pyjobshop.cp_sat`, explicit `interface_kind: offline_solver` and `required_information: full_static`; empty parameters. |
| Run | Optional `budget: {solver_time_limit_seconds: 60}` for CP; positive finite seconds, accepting fractional seconds. Omission resolves to 60. Online providers reject a solver budget. |
| Script | Each action is either `{operation_id: ..., processing_mode_id: ...}` or `{until: ...}`; mixed/unknown fields remain rejected. |

SPT selects `(nominal_ticks, operation_id, processing_mode_id)` among current legal
candidates, without intentional waiting or future information. The ft06 example
has full-static scenario visibility so it can be reused for CP, but SPT still
receives only the decision context.

`algorithms.solver` defines frozen `SolveRequest`, `ScheduleSolution`, and the
`SolverAdapter` protocol. A request contains validated factory/workload inputs,
the makespan objective, time budget and effective solver seed. The concrete
adapter maps capacity-one machines, mandatory non-preemptive tasks, one mode per
task, and explicit predecessor constraints. External task/resource/mode indices
are mapped explicitly back to semantic IDs before leaving the adapter. No
external model or mutable solver state reaches the engine.

The `cp` extra pins **PyJobShop 0.0.9 / OR-Tools 9.12.4544**. The backend is CP-SAT
with one worker. There is no parameter passthrough or dynamic provider import.
Instances with total duration greater than PyJobShop's horizon `2**42` fail
explicitly. Missing dependencies produce an error suggesting the optional extra;
`validate` does not import or execute the solver and never allocates a run directory.

Named seed derivation remains `smartsom.seed/v1`. CP consumes `solver`; generation
independently consumes `workload`. For run seed 42, the named solver seed is
**9725273414194853204** and CP-SAT receives **1017279828**, computed modulo `2**31`.
Both are recorded along with the budget and worker count. Importing an instance
never regenerates its content when the run seed or provider changes.

## Fixed external reference

The small fixtures in [`data/reference/ft06`](../../data/reference/ft06/) are
fixed snapshots, not runtime downloads or additional dependencies:

- JobShopLab commit `764af47cb5ca3ab7666d08ac8b84385207bfffd9` supplies the
  [instance](https://github.com/proto-lab-ro/jobshoplab/blob/764af47cb5ca3ab7666d08ac8b84385207bfffd9/data/jssp_instances/spec_files/ft06)
  and [start-time table](https://github.com/proto-lab-ro/jobshoplab/blob/764af47cb5ca3ab7666d08ac8b84385207bfffd9/data/jssp_instances/spec_files/solutions/ft06_sol).
  Local copies preserve the original bytes. Table rows are jobs and columns
  are **machines**, not route positions. For each route operation,
  `start = table[job][machine]`, `completion = start + duration`. The IDs are
  `ft06/job_{j}/op_{i}`, `M{machine}`, and `standard`, with zero-based indices.
- JobShopLib commit `460510f197744eed1cbcbbdfd6ec3252252f412c` supplies the ft06
  JSON entry from its benchmark dataset. All 36 machine/duration pairs match.
  The snapshot records optimum, lower bound and upper bound 55, also documented
  in the [benchmark API](https://job-shop-lib.readthedocs.io/en/stable/api/job_shop_lib.benchmarking.html).
- The [ScheduleOpt CPO record](https://raw.githubusercontent.com/ScheduleOpt/benchmarks/main/jobshop/solutions/results_cpo_22_1_1_0.json)
  independently reports 55/55 but explicitly has no certificate. Its ft06 entry,
  source URL, source-byte hash and retrieval date are saved as a bounds source,
  not as a schedule certificate.

`sources.json` contains source commit/path/byte hashes, snapshot hashes, conversion
notes and canonical workload/schedule digests. The tests independently reconstruct
all intervals from the original tables, check the domain routes, and recompute
precedence, capacity, full coverage and makespan. The reference includes real idle
time: job 0 starts its first operation at 0 and its second at 27. The new CP
solution need not match this reference's individual intervals.

## Evidence and success criteria

`run_one()` is the sole experiment entry. CP solves before entering the shared
online/replay step loop. Before schedule validation/replay, it saves
`solver_result.json` with semantic schedule, status, objective, bound, relative
gap `(objective - bound) / objective`, runtime, budget, one worker and both seeds.
The manifest includes actual PyJobShop/OR-Tools versions, input hashes, provider
identity, interface, scenario visibility, information projection and source Git
identity. Solver runtime is a measurement, not part of the semantic trace.

An `OPTIMAL` result must have matching objective and bound. A `FEASIBLE` result
may succeed after exact replay, with `proven_optimal: false`. The objective must
equal the requested schedule's maximum completion, and the actual engine result
must match every interval. Non-finite solver measurements become JSON null; no
incumbent also has a null objective/bound. Invalid output, no incumbent, provider
exceptions and replay mismatches retain failure evidence and never publish a
successful makespan. Replay failures retain the already-saved solver output and
all actual trace produced; storage failure limitations from item 2 still apply.

Determinism covers fixed input plus fixed actions/schedule, including input
reordering, working-directory changes and `PYTHONHASHSEED`. Different solver
versions or time-limit cutoffs do not promise the same selected schedule.

## Verified acceptance

- **64 focused tests passed**, covering wait semantics, canonical/invalid replay,
  SPT selection, solver contracts/configuration, solver evidence and actual CP.
- **215 tests passed in the locked CP environment**, including actual ft06
  OPTIMAL / objective 55 / bound 55 and exact replay to 55.
- **213 tests passed, 2 CP-only tests skipped in the base environment**, with both
  optional packages actually absent. The mandatory CP gate above passed all 215.
- Locked synchronization, `uv pip check`, Ruff lint/format, and `git diff --check`
  passed. Generated artifacts are excluded from the commit.
- The original 151 tests continue passing; the former unsupported-provider test
  now uses `builtin.unsupported` because SPT is implemented.
- No-incumbent, feasible-not-proven-optimal, inconsistent output, provider crash
  and replay failure evidence are exercised with controlled solver responses.
  Actual ft06 solving is separately tested with the pinned backend.

Pre-commit development smoke: SPT produced a legal makespan **88**; reference
replay produced **55**; CP produced **OPTIMAL / 55 / 55**, followed by exact engine
replay. SPT's acceptance condition is legality and complete processing, not 55.
After committing, repeat all three checks and retain their new `main` commit
identity and results under ignored local artifacts. A manifest may remain dirty
because unrelated governance/paper edits are preserved; those are not silently
included in the feature commit. No formal research study is claimed here.

The subtraction review retains only concrete policies, one solver adapter and
shared schedule replay. There is no FJSP mapping, dynamic module, batch runner,
learning framework, plugin registry or arbitrary solver override surface.
