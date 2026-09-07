# 0002 — Experiment Configuration and Execution Contract

Date: 2026-09-04

Status: Accepted

## Context

SmartSOM needs one experiment model that can represent static JSP and FJSP,
dynamic JSP and FJSP, generated workloads, imported benchmarks, online
policies, and offline solvers. The model must also support fair replications,
algorithm comparisons, and parameter sweeps without creating a second batch
execution path.

A single all-purpose experiment file would be easy to start but would duplicate
factory, workload, and algorithm definitions across studies. Allowing each
algorithm adapter or generator to invent its own configuration would also make
seeds, information assumptions, and run evidence difficult to audit.

This decision fixes the ownership boundaries and execution path. Exact typed
fields and serialization details remain implementation decisions for the
milestone that first introduces each behavior.

## Decision

### Configuration ownership

Each scientific value has one authoritative owner:

- `FactorySpec` describes static production resources, capabilities,
  calendars, buffers, and any available transport representation. Layout and
  explicit travel-time data are alternative representations; unsupported
  logistics can be omitted.
- `WorkloadProfile` describes how jobs or orders are generated, including
  distributions, ranges, correlations, and demand assumptions.
- `WorkloadInstance` is the fully materialized hierarchy described by
  `Order -> Job -> Operation -> ProcessingMode`. It is the input consumed by
  the simulator. Every processing mode owns its required resources and nominal
  duration. Matrices may be accepted as import or generator inputs, but are not
  a second runtime truth.
- `ScenarioSpec` composes a factory reference with exactly one workload source,
  selects dynamic event modules, defines information visibility, and owns the
  horizon and termination conditions. A workload source is either a profile to
  materialize or an existing instance, never both.
- `AlgorithmSpec` selects a stable provider identifier, interface kind,
  parameters, required information scope, and optional checkpoint. It does not
  contain absolute Python import paths or a per-experiment adapter definition.
- `RunSpec` binds one scenario to one algorithm and owns the objective, budget,
  root seed, telemetry policy, and output location. It is the user-authored
  specification and may contain references.
- `BatchSpec` expands replications, algorithm comparisons, and typed parameter
  sweeps into immutable plan entries. It also owns the batch root seed,
  concurrency, resume, and failure policy.

The resolver produces a `ResolvedRun`, not another authoring configuration. A
`ResolvedRun` contains normalized embedded specifications, effective seeds,
materialized input references and digests, the objective and budget, provider
identity, and output policy. It is the only input accepted by `run_one()`.

This is an experiment-layer contract. The lower-level simulation engine accepts
validated materialized domain inputs; it does not import `ResolvedRun`, resolve
file references, select an algorithm, or manage output paths. Code-constructed
fixtures and resolved experiments enter through that same domain boundary.

A normal single-run bundle therefore contains five reusable authoring files:

```text
factory.yaml
workload_profile.yaml | workload_instance.json
scenario.yaml
algorithm.yaml
run.yaml
```

The user invokes only `run.yaml`; references connect the other files. Dynamic
event declarations remain part of the scenario rather than requiring an empty
event file for static problems. A materialized event set may be referenced by a
scenario when an identical published or previously generated disturbance trace
must be replayed.

Recommended repository locations are:

```text
configs/
  factories/
  workloads/
  scenarios/
  algorithms/
  runs/
  batches/
data/
  instances/
  event_sets/
runs/
```

Configuration names use lowercase `snake_case`; stable semantic IDs inside
instances are independent of list positions and file locations.

### Materialization boundary

The formal simulator consumes only a validated, fully materialized
`WorkloadInstance` and realized event stream. Workload and event generators run
before simulation. They must record their profile, generator identity, version,
effective seeds, and output digest.

Established benchmark encodings such as `.fjs` are ingress formats. Importers
translate them into `WorkloadInstance`; they do not create parallel simulator
data models.

For online arrivals, `release_at` means the earliest physical availability and
`reveal_at` means when the algorithm may observe the job. These meanings remain
distinct even when their values are equal.

This boundary permits generated and detailed inputs to share the same engine,
makes an exact scenario replayable across algorithms, and prevents generators
from silently sampling different worlds inside algorithm runs.

Batch preparation first expands each non-algorithm parameter cell and
replication into a world identity. It materializes that world once, then gives
every paired algorithm a `ResolvedRun` referencing the same immutable instance
and event digests. Aggregation rejects paired results whose recorded world
digests differ.

### Algorithm and adapter boundary

Built-in dispatching rules implement `OnlinePolicy` directly. An online
framework adapter translates a `DecisionContext` into its external
representation and translates the result back into a `SemanticAction`. An
offline solver adapter instead accepts a `SolveRequest` containing the resolved
scenario, objective, budget, effective solver seed, and information contract,
and returns a `ScheduleSolution`. Neither adapter may mutate canonical
simulator state.

Shared code adapters are registered under stable provider IDs such as
`builtin.spt` or `sb3.maskable_ppo`. Scenario visibility defines the maximum
information the environment may expose; the algorithm declares the information
it requires. Resolution fails if the requirement is not permitted by the
scenario, and the resulting projection is recorded in the manifest. These are
compatible constraints rather than competing owners of one value.

There is no per-run `adapter.yaml`. The resolved provider implementation,
module and class identity, package versions, Git source identity, checkpoint
digest, and information assumptions are captured in the run manifest.

### Seed ownership and derivation

Numeric seed authority belongs to `RunSpec` for a standalone run. For a batch,
`BatchSpec` owns one batch root seed and each expanded `ResolvedRun` records its
derived effective seeds. Factory, workload profile, scenario, and algorithm
files may declare stochastic behavior or a seed domain, but do not embed
experiment seed values. Materialized instances and event sets retain effective
seeds as provenance, not as competing seed authorities.

The initial seed domains are:

```text
workload
demand
machine_events
processing_time
algorithm
solver
```

`workload` controls job and operation structure, routing, and nominal mode
selection. `demand` controls order or job quantities and release/reveal timing.
`processing_time` controls sampled deviation from nominal mode durations.
`machine_events` controls breakdown and repair realization. `algorithm` and
`solver` control only provider-side stochasticity.

`transport`, `quality`, and `worker_events` are added only with the corresponding
implemented behavior. The resolver uses a versioned stable derivation function;
language runtime hash functions are not used. A standalone run derives each
domain from its root seed and domain name. A batch first derives a world root
from its batch root seed, normalized non-algorithm parameter-cell identity, and
replication index. World domains derive from that world root. Algorithm and
solver domains additionally include the normalized algorithm-variant identity.

For paired algorithm comparisons, all algorithms in one world share the same
world-domain seeds and the exact same materialized inputs. Algorithm and solver
seeds remain variant-specific. Same-time event ordering is a deterministic
engine rule, not a random seed domain.

Seeds are necessary but not sufficient for reproducibility. The manifest also
records source identity, dependency versions, hardware or device information,
threading, and deterministic-mode settings where relevant.

### Execution path

The planned command surface is:

```text
smartsom validate CONFIG
smartsom plan CONFIG
smartsom run RUN_CONFIG
smartsom batch BATCH_CONFIG
```

`validate` performs typed and cross-reference validation. `plan` resolves a run
or expands a batch without persisting artifacts or simulating, showing the run
count, effective seeds, and configuration differences. `run` resolves and
materializes a `RunSpec`, then invokes `run_one(ResolvedRun)`. `batch`
materializes each world once, writes the immutable expanded run plan, and
invokes that same `run_one()` function for each child. Directory scanning is
not an experiment definition.

Scientific grids and replications belong in `BatchSpec`. The planned common
operational overrides are `--output-root`, `--log-level`, `--seed`, and an
explicit typed `--set path=value` for exploratory use. Batch execution also
accepts `--workers`, `--resume`, and `--fail-fast`. Every override must be
persisted in the resolved configuration and manifest.

Sweep dimensions and their values are ordered lists. Their default expansion is
the Cartesian product; a `zip` mode must be explicit and requires equal-length
dimensions. A plan entry may not assign the same resolved path from multiple
dimensions. Algorithms retain declaration order and replications use ascending
indices, giving deterministic plan order.

Each child receives a deterministic `plan_entry_id` derived from normalized
scientific inputs, the algorithm variant, and replication index. Duplicate IDs
are validation errors. Resume matches the plan entry ID and resolved input
digests, never a directory position. The human-facing run-directory naming
scheme remains deferred.

### Run evidence

A run directory is allocated only after authoring and cross-reference
validation succeeds. From allocation onward, every attempted run retains its
resolved run, manifest, progress, terminal summary, and structured failure when
applicable. The manifest begins with the attempt and is finalized with status
and all available digests. The materialized instance, realized events, semantic
trace, and metrics are required only after the lifecycle stage that can produce
them has succeeded or begun.

A batch additionally persists its resolved batch definition, immutable expanded
run plan, aggregate summary, and child run directories. Invalid batches fail
before the run plan or child directories are created.

Local artifacts remain authoritative. External tracking systems may mirror
them, but cannot replace the local evidence or provenance record.

## Consequences

- JSP is represented as an FJSP instance with one processing mode per
  operation; static problems omit dynamic event modules. No separate engine or
  file family is required for each problem label.
- Factories, workloads, and algorithms can be reused independently across
  experiments without duplicating values.
- Generated studies incur a materialization step and additional provenance
  artifacts before simulation.
- Batch planning is explicit and auditable, but very large sweeps may produce
  large run plans and must later be streamed rather than held entirely in
  memory.
- Exact field serialization, human-facing run-directory naming, and distributed
  execution remain deferred until their first implementation milestone.

## Rejected Alternatives

- A monolithic experiment file: simple initially, but poor for reuse and value
  ownership.
- Separate adapter and seed files: they add indirection without creating an
  independent reusable scientific object.
- Absolute class or source paths in algorithm configuration: they make
  configuration environment-specific and bypass provider validation.
- Sampling workloads or disturbances inside the simulator loop: it prevents
  exact paired comparisons and obscures provenance.
- Separate handwritten scripts for single runs, replications, and sweeps: they
  create multiple execution semantics and incomplete evidence.
