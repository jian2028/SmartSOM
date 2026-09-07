# Week2 item 6 — Materialized processing-time uncertainty

This slice preserves nominal workload content and supplies a separate immutable
`ProcessingTimePlan` for execution. It supports fixed input, seeded independent
operation-mode multipliers, online policies, exact replay and a small composition
with arrivals. It does not add dependencies, stochastic CP, correlated modes,
breakdowns, learning, batch or additional probability distributions.

## Public contract

```python
from smartsom.domain import ProcessingTime, ProcessingTimePlan
from smartsom.engine import Simulator, replay, replay_schedule

# Include every operation-mode in the actual workload, not only chosen modes.
times = ProcessingTimePlan((ProcessingTime("A1", "standard", 10, 12),))
simulator = Simulator(factory, workload, processing_times=times)
```

`processing_times=None` preserves nominal execution. The plan has a `modes`
tuple sorted by `(operation_id, processing_mode_id)`; each entry records nominal
and actual positive integer ticks. It must cover the workload exactly, and its
nominal values must match that workload. Booleans, floating durations, missing,
duplicate and unknown IDs fail before simulation. Mode IDs are operation-local;
several alternatives on the same machine remain distinct.

`replay` and `replay_schedule` accept the same `processing_times` keyword, along
with the existing arrivals and trigger keywords. All execution uses `step()`.
Completion events, machine occupancy intervals, schedule validation and runtime
invariants use actual duration; public candidates and SPT use nominal duration.

Before completion, observations expose neither actual duration nor the planned
completion timestamp or real remaining duration. After completion, existing
start/completion fields disclose only the selected mode's realized duration.
Unselected alternatives remain private. Elapsed time and the fact that a task
has not completed are naturally observable; this is not a promise to conceal
information logically inferable from public execution history.

No trace field is added: dispatch retains nominal duration, and completion
records the actual event time. Turning the module off or using unit multipliers
preserves the entire original static trace. Other constant multipliers are
deterministic scaled execution. The existing CP provider rejects any enabled
processing-time declaration, including unit multipliers and nominal fixed tables;
`full_static` visibility is also rejected. Exact schedule replay is a privileged
validation path, not an online policy or oracle solver provider.

## Configuration and exact sampler

The scenario is the only enabling authority. Omit/null `processing_time` to
disable it, or select one of these declarations:

```yaml
processing_time:
  kind: uniform_multiplier
  profile: {low: 0.8, high: 1.2}
```

```yaml
processing_time:
  kind: fixed
  path: ../../data/processing_times/hand.json
```

The optional profile defaults to 0.8/1.2. Bounds accept decimal numbers or
decimal strings and enforce finite `0 < low <= high`. Strings retain precision
beyond a JSON/YAML floating-point number; equivalent decimal spellings normalize
to the same profile digest. Unsupported fields such as runtime sample timing,
seed overrides or different sampling units fail validation.

Generator version `smartsom.processing-time/v1` fixes the following recipe:

1. Take the run's existing derived `processing_time` unsigned 64-bit seed.
2. Encode `[version, seed, operation_id, processing_mode_id]` as compact UTF-8
   JSON with Unicode preserved. Hash with SHA-256 and interpret all 32 bytes as
   an unsigned big-endian integer.
3. Construct a local `random.Random` from that integer and draw `getrandbits(53)`.
   Define `u = draw / 2**53` and multiplier `low + (high-low)*u` using exact
   fractions of the decimal bounds. This is a uniform 53-bit grid in `[0,1)`.
4. Compute `max(1, floor(nominal * multiplier + 1/2))` exactly, without Python's
   bankers rounding, floating-point approximation or ambient Decimal context.

Every mode has an independent identity, including equal nominal times on the
same machine. Reordering or adding other modes/operations does not move its
draw. Sampling occurs only during resolution, before directory allocation.
Off, fixed and constant-profile paths do not draw randomness; constants record
null draws. Other named-seed values and consumption flags remain unchanged.

Golden root seed 42 gives processing seed **3797569027003476775**. For standard
modes A1/A2/B1/B2, the fixed draws are **7221518816510873, 8380621278969816,
92466246856231, 8063507948353621**. With nominal **10/5/5/10**, the default profile
produces actual **11/6/4/12** and the example SPT schedule has makespan **23**.

## Fixed inputs and evidence

`smartsom.processing-times/v1` JSON contains `processing_times: {modes: [...]}`,
optional `content_sha256`, and optional `provenance`. Each mode row names both
semantic IDs and its `nominal_ticks`/`actual_ticks`. Handwritten fixed inputs
need no fictitious random samples.

Generated provenance includes generator/version, normalized profile and digest,
effective seed, and semantic-ID draw rows. Draws plus bounds represent the exact
multiplier without a lossy decimal approximation. Import checks coverage,
declared digests and the arithmetic relationship between supplied draws and
actual durations. It does not regenerate draws from historical seeds. Historical
provenance is a recorded claim, not authentication of an externally supplied file.

Enabled runs save reusable `realized_processing_times.json` before simulation
and the exact delivered `observations.jsonl` before each policy call. Arrival
compositions share the observation file. Manifest records separate workload,
processing-time content, original source-byte and profile digests, plus generation
provenance. Changing seed/provider after importing the fixed files changes neither
world input. Failed runs preserve inputs, delivered snapshots and partial trace,
without a successful makespan. These private evidence files are never policy input.

## Independent reference and statistical checks

| Operation | Machine | Nominal | Actual | Reference interval |
|---|---|---|---|---|
| A1 | M1 | 10 | 12 | [0,12) |
| B1 | M2 | 5 | 6 | [0,6) |
| A2, after A1 | M2 | 5 | 4 | [12,16) |
| B2, after B1 | M1 | 10 | 8 | [12,20) |

Makespan is **20**. The fixed [JobShopLib source record](../../data/reference/processing_times/sources.json)
uses commit `460510f197744eed1cbcbbdfd6ec3252252f412c`; the adjacent reproduction
script injects actual durations directly into a two-job deterministic instance.
Source/result hashes, raw solver output and a semantic schedule are retained.
No JobShopLib dependency is required for ordinary tests. This source validates
the realized execution, not our sampler. JobShopLab's offset-and-truncate model
is not a numerical oracle for this multiplier/half-up contract.

Fixed-draw checks independently use high-precision Decimal half-up arithmetic,
including an exact half boundary, a value just below it, and the minimum-one clamp.
The statistical gate pre-registers 10,000 samples per combination of nominal
`1/10/100` and ranges `[0.8,1.2]`, `[0.9,1.4]`, `[0.1,0.3]`. Independent bin
integration computes the rounded/clamped theoretical mean. The tolerance is
`(max_actual-min_actual)*sqrt(log(2*9/1e-6)/(2*10000)) + 1e-10`, giving a
Hoeffding union-bound error budget of `1e-6` over all nine cases and covering the
negligible 53-bit discretization difference. This is an engineering sampler
check, not a research performance result; asymmetric/clamped means need not be
nominal, and tiny nominal times can legitimately lose all variation after rounding.

## Commands and completion evidence

```bash
uv run smartsom validate configs/runs/processing_generated.yaml
uv run smartsom run configs/runs/processing_fixed.yaml       # 20
uv run smartsom run configs/runs/processing_generated.yaml   # 23
uv run smartsom run configs/runs/processing_arrivals_dispatch.yaml
uv run smartsom run configs/runs/processing_arrivals_event.yaml
```

Run focused processing tests first, then the actual locked base environment and
the locked CP environment with `SMARTSOM_REQUIRE_CP=1`. Retain all earlier
regressions, including ft06 55, official FJSP 6, Mk01 40, and arrival evidence.
Check dependency consistency, Ruff, formatting and `git diff --check` before
staging only this slice. Post-commit results and source identity are saved under
ignored `artifacts/week2-item6/`; generated run directories are not committed.

Pre-commit verification on 2026-09-07: **80 focused processing tests passed**;
the actual base environment returned **434 passed, 7 optional CP skips**, with
PyJobShop/OR-Tools/fjsplib absent. The locked CP environment returned **441 passed**,
including all seven real solver regressions. Both environments pass dependency
consistency checks. Ruff, formatting and diff checks are required before commit.
