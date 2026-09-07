# Refactoring before Week2 item 7

This change preserves the item 1-6 behavior. It does not implement breakdowns,
batch execution, new logging controls, or bounded-memory simulation.

## Engine and incremental trace reads

Operation release times are indexed once in an immutable mapping. Invariants
reuse a single pending-calendar snapshot per check and the immutable set of
arrival events. All global checks and their execution frequency remain intact.
`Simulator.trace_since(cursor)` returns an immutable, non-consuming suffix;
the existing full trace and `SimulationResult` are unchanged.

The configured-run comparison against `f4bf22e4308b5e27a583eb1c6d6159a67be0f7bc`
covers all 17 online run presets. Normalized resolved inputs (excluding only the
operational output root), every named seed, complete simulation results,
observations, metrics, progress, realized inputs and semantic manifest fields
match exactly. Source metadata, attempt paths and resulting manifest hashes
naturally differ. CP is checked separately with actual solving and exact replay;
solver-generated schedules are not used as cross-version golden records.

### Local diagnostic

Five repetitions per size, eight machines, five operations per job, two eligible
machines per operation, nominal ticks in `[1, 5]`, `static_fjsp_v1` seed 101 and
SPT. Times include simulator construction and execution, excluding generation
and serialization. Peak allocations come from a separate `tracemalloc` pass,
not process RSS. This is a local engineering diagnostic, not a throughput claim
or CI timing threshold.

| Operations | Baseline median (s) | Indexed median (s) | Baseline / indexed peak bytes | Full observation bytes |
| --- | --- | --- | --- | --- |
| 50 | 0.008941 | 0.005393 | 93,424 / 88,344 | 983,392 |
| 100 | 0.039653 | 0.018569 | 230,012 / 232,892 | 3,909,804 |
| 200 | 0.209147 | 0.068798 | 472,972 / 473,524 | 15,572,023 |

For 200 operations, profiled calls fell from about 7.24 million to 1.57 million.
The repeated release scan is no longer a dominant cost. Full invariants remain
the main cost, and memory and observation volume are not materially reduced.
Observation sizes measure every public decision serialized as currently written
when observation recording is enabled; the timing case itself is static.

Local raw reports and the comparison driver are under the ignored
`artifacts/pre-item7-refactor/`. The complete existing tests additionally cover
reordered inputs, hash seeds, both arrival triggers, hidden information, module
disablement, semantic replay and retained failures.

The engine checkpoint passes 444 base-environment tests (seven optional CP
tests skipped only there), with the absence of PyJobShop, OR-Tools and fjsplib
verified. The locked CP environment runs all solver acceptances. Ruff, formatting,
dependency consistency and whitespace checks are also required before commit.

## Experiment preparation and evidence

The second checkpoint separates already-parsed materialization from reference
resolution and algorithm binding. Frozen internal results hold realized inputs,
provenance and consumption flags; `ResolvedRun` keeps its existing public fields.
All four providers have explicit construction, with no fallback for an unknown
provider. Compatibility and script-reference checks retain their pre-execution
failure behavior.

`RunEvidence` receives records and results; it owns files and derived counters,
not simulator state. The runner still invokes the same public step loop and
saves solver output before schedule validation/replay. File hashing now uses
bounded reads. No file schemas, record verbosity, flush frequency, CLI flags or
optional dependencies changed.

Validation on this checkpoint:

- 177 focused preparation, run-evidence, solver and dynamic-input tests passed.
- 450 base-environment tests passed; seven optional CP tests were skipped only
  in that environment, where solver packages are absent.
- 457 tests passed in the locked CP environment, including actual ft06=55,
  official FJSP=6 and Mk01=40 optimality/bound/replay checks.
- All 17 configured online cases still match the pre-refactor baseline in
  resolved inputs, complete results and existing evidence bytes, with only the
  operational/source exceptions described above.
- New checks cover unchanged materialized components across module switches and
  algorithm changes, explicit provider rejection, streamed file hashes, and
  trace/metrics/observation writer failures preserving original causes, actual
  completion counts, partial records and failed summaries without makespan.
- Ruff, formatting, both environments' dependency checks and `git diff --check`
  passed. The lockfile and authoring presets are unchanged.

The local second-checkpoint 200-operation median was 0.073407 s, with the same
15,572,023 observation bytes and complete result digest. Full audits, snapshots
and in-memory trace retention remain costs to measure before larger workloads.

## Deferred work and reusable follow-up prompts

These are future requests, not implementation claims or automatic execution.

### Item 7: actual breakdown semantics

Begin Week2 item 7 by checking the current state/event contract. Implement
breakdown/repair with processed/remaining-work conservation, same-machine
pause/resume and invalidated completion handling. Keep the engine's exclusive
state ownership, disabled-module equivalence and deterministic replay. Discuss
unresolved semantics before implementation; do not prebuild generic hooks.

### After item 7, before logistics

Reconcile the Week2 AGV-first list and roadmap's buffer-first direction before
starting logistics. Define ownership of occupancy, availability, job location,
transport and buffer reservations using the actual next feature. Extract only
shared responsibilities with real users, and test composition and disablement.

### After item 10, before item 11 comparisons

Implement study/batch on a local machine or one Linux server through `run_one()`.
Support case/algorithm/replication composition, paired ablations, plan preview,
bounded process concurrency, retained failures and input-identity-based resume.
The study owns the root seed and generates child runs without handwritten YAML
per combination. Separate single-run and batch progress, scientific evidence
and default-off, size-limited debug diagnostics. Show solver stage/elapsed time
without inventing a completion percentage. Validate on small combinations first.

### After item 11, before learning adapters

Design observation/action translation, semantic action mapping and masks against
the stable engine. Select actual frameworks and algorithms in algorithm presets
and keep dependencies optional. Measure environment stepping, projection,
evidence and training throughput before choosing incremental candidates, lighter
recording or batched environments. Preserve simulation truth and visibility.
