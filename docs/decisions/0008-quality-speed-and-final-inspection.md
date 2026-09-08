# 0008 — Quality-speed capabilities and final inspection

Date: 2026-09-08

Status: Accepted

## Decision

Factory capabilities own a shared quality-mode table and optional complete
per-machine replacement tables. A mode has a semantic ID, positive finite decimal
time scale and finite decimal error probability in [0,1]. Tables need not be
monotone; equal-duration and dominated choices remain distinct. Every base mode
used by the workload must resolve a nonempty table. There is no operation-level
override, partial field merge or implicit IDETC preset in the engine.

The scenario alone enables quality, selects generated or fixed draws, and chooses
`probability_visibility: public | hidden` (default public). Algorithm parameters
choose free selection or one fixed quality label. Numeric seeds remain run-owned.
Factory declarations remain inactive when scenario quality is absent/null.

Base workload and UPT inputs remain authoritative and unchanged. The pure
`prepare_quality(factory, workload, draws, processing_times=...)` produces an
immutable `QualityPlan`: operation draws and a complete execution-mode catalog.
`QualityModule` validates that plan against its original inputs before projecting
execution-only workload and processing-time tables. Simulator and timetable replay
use the same projection. Persisted base inputs can be reimported without scaling
them a second time.

Every base mode is crossed with its machine's effective quality table. Full mode
IDs are `quality-v1/<escaped base ID>/<escaped quality ID>`, with each component
UTF-8 percent-encoded using `urllib.parse.quote(..., safe="")`. IDs are unique
within an operation; operation IDs remain unchanged. Explicit catalog fields retain
base ID, quality ID, machine, scale, probability and execution durations.
`Dispatch` still chooses one semantic processing-mode ID at actual start, and that
choice remains fixed through pause/resume and completion.

Both nominal and actual execution durations use exact rational half-up rounding,
with minimum one tick. Actual duration first uses the existing base UPT integer
realization, then applies the selected quality scale. The baseline nominal value
is scaled separately for policy observation. Outage, queue, blocking and travel
intervals are never scaled. Distinct modes may round to the same duration.

## Randomness and state

Enabled quality appends a `quality` seed using the unchanged `smartsom.seed/v1`
root/domain recipe. Existing six values are unchanged; quality-off runs retain
the original six-entry seed table. Fixed quality imports mark the new seed unused.
Generated quality always materializes an operation draw, including probabilities
zero and one, so the latent input can be reused under another probability table.

For each operation, compact UTF-8 JSON of
`["smartsom.quality/v1", effective_quality_seed, operation_id]` is SHA-256 hashed.
The full digest interpreted as an unsigned big-endian integer seeds a local
`random.Random`, whose `getrandbits(53)` supplies the draw. There is no shared RNG,
sequential stream or runtime sampling. Every machine and quality alternative of
one operation shares `u = draw / 2**53`; different operations have independent
identity-keyed draws. An error occurs exactly when `u < error_rate`.

The engine evaluates each operation once on its real processing completion,
records the chosen mode/outcome, and permanently marks its job defective on the
first error. Later operations are still processed and checked, including after an
earlier error. Pauses, repairs, blocked holding and deliveries do not repeat a
check. There is no rework, early scrap or quality-dependent resource transition.

Quality checks follow their completion record within the existing completion
phase. Physical event ordering and the handoff closure remain unchanged. Final
inspection follows actual output delivery/Transfer, including automatic unloads
at the end of a tick. With both logistics switches off it follows the last
operation's completion. Inspection has zero duration and no resource/action.
There is no new decision trigger. A run terminates when the existing production
and logistics criteria are met, regardless of how many jobs fail inspection.

## Information and evidence

Visible operations have immutable quality-mode views with semantic identities,
machine, scale and effective nominal duration. Hidden probability uses `null`,
never a fabricated zero; the numerical probability is available only in public
mode. Hidden jobs are absent from every quality view. Job inspection is unknown
until output, then exposes only job pass/fail and inspection tick. Per-operation
outcomes and draws never enter policy observations, even after inspection. UPT's
unfinished actual durations stay private. Hiding a field does not erase any prior
knowledge a policy may have about named presets.

`quality_check` and `inspection` are audit trace records, not actions. Privileged
run evidence contains draws/provenance, effective mode catalog and full trace;
only `DecisionContext` is supplied to policies. A probability visibility switch
changes observations, not world digests or physics for a fixed action sequence.

A completed result includes operation outcomes, job inspections and equal-job
passing rate: passed jobs divided by all jobs. Failed attempts retain partial
records but never publish a terminal passing rate or successful makespan. Quality
failures alone are successful simulator executions with nonconforming products.

SPT and first-feasible support a fixed label only when all candidate base modes
support it; resolver failure replaces neither routes nor labels. SPT otherwise
uses scaled nominal duration and the existing semantic tie-breakers. CP rejects
quality-enabled runs, even with zero probabilities. No quality objective, learned
reward, framework, batch interface or dependency is introduced.

## Compatibility and evidence boundary

This extends ADR 0002's capability/input/seed boundary and ADR 0004's independent
paired inputs. It does not revise the earlier event and logistics contracts.
Quality-off traces, observations, schedules and generators retain their behavior.
Enabling a zero-probability table still emits quality evidence.

Draft4 page 4 supports the three example parameter pairs and permanent job defect
rule. The explicit RNG coupling, rounding, final-only observation and configurable
machine overrides are this project's agreed contracts, not claims about omitted
paper implementation details. See [acceptance](../validation/quality-speed.md).
