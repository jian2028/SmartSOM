# SmartSOM Roadmap

Status: M0 foundation, the validated static serial FJSP core of M1/M2, and the
single-run config/import/generation/evidence, SPT/CP and exact schedule replay
subsets of M3, plus online arrivals and processing uncertainty from M4. General algorithms, batch
experiments and the remaining dynamic milestones are incomplete.

Each milestone is a bounded vertical slice. A later milestone does not begin
merely because its directories or interfaces have been discussed.

## Next implementation slices

The immediate Week 2 sequence narrows the broader milestones below:

1. **Implemented and validated:** the static core subset of M1/M2 has separate factory/workload inputs,
   serial job chains, one mode per operation, semantic step/run/replay, integer
   completion events, immutable decisions, trace, makespan, and invariants.
   Resource competition and crossing routes are checked with two hand-computable
   fixtures. No config, CLI, dynamic module, solver, or framework is included.
   See the [static core validation record](validation/static-core.md).
2. **Implemented and validated:** the five-file config/resolver/evidence slice
   through `run_one(ResolvedRun)`, with strict validation, named seeds, a static
   JSP profile generator, reusable imported instances, two toy policies, minimal
   CLI, and successful/failed run evidence. The two code-built hand fixtures
   retain identical schedules, traces, and makespans through configuration.
   See the [configured-run validation record](validation/configured-runs.md).
3. **Implemented and validated:** explicit waiting, full semantic schedule replay,
   SPT and an optional pinned PyJobShop/CP-SAT adapter. A fixed ft06 reference
   replays exactly to 55; actual CP returns OPTIMAL / 55 / 55 and its output also
   replays exactly. Feasible incumbents and proof of optimality remain distinct,
   with failed solver/replay attempts retained. See the
   [static JSP validation record](validation/static-jsp.md).
4. **Implemented and validated:** multiple modes, selected-mode runtime state,
   traditional `.fjs` import, independent seeded FJSP generation, and multi-mode
   CP mapping. Hand paths replay to 4 and 7. The official PyJobShop example and
   Mk01 have fixed references and actual adapter solutions replaying to 6 and 40;
   ft06 retains 55. The base environment supports all non-solver paths. CP runtime
   classification is deferred while raw solve metrics remain recorded. See the
   [static FJSP validation record](validation/static-fjsp.md).

5. **Implemented and validated:** independent release/reveal plans, filtered full-job
   observations, two explicit decision triggers, event waiting, fixed timing import,
   seeded arrival generation and observation/evidence persistence. The release-
   constrained hand/reference schedule replays to 6; all-zero plans preserve the
   old complete trace. No dynamic CP, breakdown/repair, batch or learning is added.
   See the [online-arrival validation record](validation/online-arrivals.md).

6. **Implemented and validated:** independent actual mode durations, fixed import,
   semantic-ID multiplier sampling and exact half-up rounding. Online decisions
   retain nominal information; completion reveals only the executed mode's time.
   The fixed external/hand schedule replays to 20, unit multipliers preserve old
   traces, and micro cases compose with arrivals. CP remains static-only.
   See the [processing-time validation record](validation/processing-times.md).

The first slice does not complete the full FJSP domain or experiment milestones.
Their wider capabilities remain planned until separately implemented and tested.

Before item 7, the existing behavior is preserved while indexing release times,
adding incremental trace reads, separating parsed-input materialization and
algorithm binding/construction, and extracting run evidence management. This
does not add batch or dynamic behavior; see the
[refactoring validation and follow-up gates](validation/pre-item7-refactor.md).

Follow-up timing:

- Item 7 introduces actual pause/resume and completion-event lifecycle changes
  with breakdown/repair, rather than prebuilding a generic event framework.
- Before logistics, reconcile the older Week2 AGV-first list with this roadmap's
  buffer-first direction, then establish real resource/constraint ownership.
- After item 10 and before item 11's integrated comparisons, introduce study/batch
  composition, paired ablations, single-host process concurrency and recovery,
  and separate progress/debug/evidence policies. Use small acceptance cases first.
- After item 11, add real learning adapters in items 12-13 and profile environment,
  projection, evidence and training costs before further optimization.

## M0 — Repository Foundation

Establish an installable Python 3.12 package, locked `uv` environment, quality
gate, concise project governance, architecture contract, and roadmap. No
simulator behavior is included.

Exit criteria:

- the package installs and imports;
- Ruff and pytest pass locally and in CI;
- at foundation creation, the README states that the repository is scaffold-only;
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
