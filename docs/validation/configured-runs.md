# Configured static runs — Week 2 item 2

This record covers configuration, generation and single-run evidence on top of
the [static core](static-core.md). It establishes engineering behavior only;
SPT/CP, benchmark optimality, flexible modes, intentional waiting, dynamic events,
runtime budgets, batch execution and learning are outside this slice.
The subsequent [item 3 record](static-jsp.md) extends these v1 contracts with
waiting, SPT, full-static CP visibility, and solver budgets; the statements below
describe the item 2 checkpoint.

## Reproduce

```bash
uv sync --locked
uv run pytest -q tests/unit/test_experiments.py
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
git diff --check
uv run smartsom validate configs/runs/generated.yaml
uv run smartsom run configs/runs/competition.yaml
uv run smartsom run configs/runs/crossing.yaml
uv run smartsom run configs/runs/generated.yaml
```

`validate` resolves and materializes in memory without creating a run directory
or constructing a simulator. Each `run` allocates a new directory under the run
file's `output_root`, using UTC time plus a UUID. It never overwrites an attempt.
CLI exits are 0 for success, 2 for invalid configuration/command syntax, and 1
for execution or output errors. `RunFailedError` exposes `run_dir` and `cause` to
Python callers; the original exception is also chained.

## Five authoring contracts

Every document requires its string `schema` identifier. Unknown fields, duplicate
mapping keys, coercion from strings/bools/floats to integers, YAML object tags and
YAML merge keys are rejected. YAML/JSON is the file encoding; the scenario's
explicit source kind selects the workload schema. There is no filename-based
profile/instance inference or environment-variable interpolation.

| Owner | Required content | Supported defaults |
| --- | --- | --- |
| `smartsom.factory/v1` | `factory: {machines: [{machine_id: ...}]}` | Machines have capacity 1 implicitly; no configurable capacity field. |
| `smartsom.workload-profile/v1` | `generator: static_jsp_v1`, `profile` below | None. |
| `smartsom.workload-instance/v1` | `workload: {orders: [...]}` using the existing domain hierarchy | Optional `content_sha256` and `provenance`; these describe materialized data. |
| `smartsom.scenario/v1` | `factory` path, `workload: {kind: profile\|instance, path: ...}` | `modules: []`, `visibility: decision_context`, `termination: all_jobs_complete`; these are the only supported values. |
| `smartsom.algorithm/v1` | `algorithm: {provider: ..., ...}` | `interface_kind: online_policy`, `required_information: decision_context`. |
| `smartsom.run/v1` | `scenario` path, `algorithm` path, `seed`, `output_root` | `objective: makespan`, the only supported objective. |

The two workload schemas are alternatives, yielding five files per run. Paths
are resolved relative to the document containing them, including `output_root`
relative to the run file. Examples reuse one factory across all three runs.

`builtin.scripted` requires `parameters.actions`, an ordered list of
`{operation_id, processing_mode_id}` mappings. IDs must exist at resolution;
action order and legality are checked by the engine during execution. The runner
rejects both exhaustion before completion and unused actions after completion.
`builtin.first_feasible` accepts only empty parameters (omittable), and chooses
the minimum `(operation_id, processing_mode_id)` among current legal actions.
Neither provider uses a numeric algorithm seed or performs intentional waiting.

Strict Pydantic envelopes validate the existing frozen standard-library domain
dataclasses, without duplicating their serial-chain validation or introducing
framework imports into `domain`, `dispatch`, `engine`, or `trace`. Parsed
containers are immutable tuples. The resolver canonicalizes factory machines,
orders, jobs, and operations by their semantic IDs; predecessor IDs still define
process order. Original file byte hashes remain distinct from domain hashes.

## Generator and seeds

The complete generation profile is:

```yaml
schema: smartsom.workload-profile/v1
generator: static_jsp_v1
profile:
  order_count: 1
  jobs_per_order: 2
  operations_per_job: {min: 1, max: 2}
  nominal_ticks: {min: 1, max: 5}
```

Counts and inclusive range bounds are positive integers; `min <= max` and
`operations_per_job.max <= machine_count`. Every job has a serial route with no
repeated machine and exactly one `standard` mode per operation. All jobs are
available at tick zero. Nominal durations are sampled before simulation; they
are fixed durations, not runtime uncertainty.

The root run seed is an unsigned 64-bit integer. For each domain, v1 takes the
first eight SHA-256 digest bytes, interpreted as an unsigned big-endian integer,
of this UTF-8 string (`\0` means one NUL byte):

```text
smartsom.seed/v1\0{root_seed_in_decimal}\0{domain}
```

The domains are `workload`, `demand`, `machine_events`, `processing_time`,
`algorithm`, and `solver`. Only `workload` is consumed for generated static inputs;
all are inactive for imported inputs. Historical generation seeds on an instance
are provenance, not overrides of the current run's seed plan.

Generation v1 uses a local Python `random.Random(workload_seed)`, with sorted
machine IDs. For each numerically ordered order/job it draws the operation count
with `randint`, the route with `sample` without replacement, and each nominal
duration with `randint`. Fixed bounds still invoke `randint`. IDs are
`order_N/job_N/op_N` (job numbers local to an order; operation numbers local to a
job), and explicit predecessor IDs connect the chain. Python is pinned to 3.12;
generator version and actual Python/package versions are recorded. Reproduction
across a future generator/runtime version requires checking its golden fixtures
or importing the saved instance; seed alone is not a cross-version guarantee.

For root seed 42, the workload seed is **6941565647359864201**. The example yields:

| Job | Serial route with nominal ticks |
| --- | --- |
| `order_1/job_1` | `M2/5 -> M1/3` |
| `order_1/job_2` | `M1/1` |

Its canonical workload SHA-256 is
`c039e0307dc2c71cc3b29d6e1ee3ad466dd7055b44422a2fd254d0678d0ee7a5`.
With `builtin.first_feasible`, job 1 uses M2 at `[0,5)`, job 2 uses M1 at
`[0,1)`, then job 1 uses M1 at `[5,8)`: makespan **8**.

The generated instance envelope stores generator ID/version, profile SHA-256,
effective workload seed and canonical content SHA-256. Canonical JSON is UTF-8,
sorted object keys, compact separators, literal Unicode, no non-finite numbers,
and normalized domain collections. Paths, wall-clock timestamps and source file
formatting are excluded. For another algorithm on exactly this world, set the
scenario workload to `{kind: instance, path: .../realized_instance.json}`. Import
checks any supplied content digest, retains provenance and never regenerates.

## Evidence and failure contract

Resolution produces a frozen `ResolvedRun` with normalized embedded inputs,
absolute reference paths, source-byte hashes, canonical factory/workload hashes,
seeds and generation provenance. `run_one()` consumes this snapshot without
rereading authoring files; it creates a fresh simulator and provider each time.

Successful attempts contain `resolved_run.yaml`, `realized_instance.json`,
`manifest.json`, `progress.log`, `trace.jsonl`, `metrics.jsonl`, and `summary.json`.
The manifest binds the provider implementation and current decision projection
to the source checkout's Git commit/dirty status, Python/platform/package
versions, seeds, input hashes and artifact hashes. It excludes its own digest.
Generated evidence is ignored by Git. Dirty development runs accurately record
dirty source; they are not promoted to formal experiment evidence.

The runner calls public `step()` and flushes new trace records after every call,
including failures. Metrics contain cumulative completed-operation counts at
each actual completion, then terminal makespan on success. Trace retains the
core's semantics exactly; progress is a separate human-readable step/lifecycle
log. The CLI prints terminal status and the output path.

Input errors fail before allocation. After allocation, ordinary execution
failures retain resolved inputs, manifest, progress, `summary.json`, and a
structured `failure.json` with stage, exception type/message, simulation time and
rejected action where applicable. Trace and metrics exist once simulation starts.
Failed summaries use `makespan: null`, even if an overlong script was detected
after all operations completed. Output-storage failures are reported with the
original error and any failure-writing error; unavailable storage cannot promise
complete evidence. Abrupt process kill/power loss recovery is not implemented.

## Acceptance coverage

- Code-built and imported competition/crossing fixtures have identical domain
  objects, entire traces, schedules and metrics: makespans **6** and **5**.
- Golden seed and generator outputs, integer boundaries, route uniqueness,
  serial dependencies, global-RNG isolation and independent schedule validation.
- Different `PYTHONHASHSEED` values and working directories preserve results;
  machine/container reordering preserves semantic inputs and hashes.
- Instance export/import with a new seed/provider preserves the world and replay;
  repeated runs use new directories and produce identical semantic evidence.
- Strict schemas, references, duplicate keys/IDs, capacity, durations, ranges,
  source type, unsupported capabilities and misplaced seeds fail before engine
  construction. Existing core tests continue covering all domain invariants.
- Script exhaustion/excess/illegal actions and unexpected provider failures
  preserve partial evidence; initialization failures retain stage-appropriate
  artifacts. CLI return codes and read-only validation are exercised.
- Immutable snapshots and detached serialization prevent caller mutation;
  deleting source configs after resolution does not affect execution. The runner
  uses the public step API, and the core's dependency-isolation test still applies.

Validation for this implementation: **61 focused tests and 151 total tests
passed**, along with locked `uv` synchronization, Ruff lint/format checks and
`git diff --check`. The post-commit example smoke records its source commit in
each local manifest. Older worktree evidence is not used for this slice.
