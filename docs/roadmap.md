# SmartSOM Roadmap

Status: M0 foundation, the validated static serial FJSP core of M1/M2, and the
single-run config/import/generation/evidence, SPT/CP and exact schedule replay
subsets of M3, plus online arrivals, processing uncertainty and machine outages from M4,
and fixed-matrix transport with optional finite buffers from M5. General algorithms
experiments and the remaining dynamic milestones are incomplete.

Each milestone is a bounded vertical slice. A later milestone does not begin
merely because its directories or interfaces have been discussed.

The approved usability refactor is tracked in [its implementation record](implementation-usability.md).
Typed API/CLI, scenario authoring, single-environment full recovery and local
report/export flows have development integration evidence at local `3bb316e`.
Parallel sampling, research extensions, search and fresh macOS formal acceptance
remain in progress. This record does not replace historical item 12/13 evidence.

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

7. **Implemented and validated:** immutable fixed or seeded per-machine outage
   plans, original-machine pause/resume, cancelled completion lifecycle, current
   availability and completed net-work observations, and exact replay. Independent
   fixed-break references give 7/9/5 and the PyJobShop example gives 7; bounded
   DynaSchedBench comparisons document their redispatch and raw-Gantt differences.
   Arrivals, processing uncertainty and multiple modes compose. CP stays static-only.
   See the [machine-event validation record](validation/machine-events.md).

8. **Implemented and validated:** fixed directed-matrix AGVs, input/prebuffer/processing/
   postbuffer/output locations, explicit queue rerouting, mode selection at processing
   dispatch, complete transportation schedules and exact replay. The hand case and
   actual JobShopLab transport phases end at 15; its final-clock discrepancy is
   preserved. All 8 JA/MB/UPT combinations are covered. See the
   [transport validation record](validation/transport.md).

9. **Implemented and validated:** independent zero/finite/infinite pre/post capacities,
   exclusive AGV reservations, loaded waiting, direct completed-job pickup and
   automatic unblocking. With AGV off, explicit instantaneous Transfer uses the same
   logistics ownership. V2 execution replay preserves timed action order and every
   arrival/unload/transfer. Three hand cases give 6/8/6; independent zero-buffer CP
   holds A1 until 5 despite net completion at 2. JobShopLab capacity/flex checks
   retain its full/zero postbuffer errors as differences. Item 8 unlimited traces
   and both movement modes' JA/MB/UPT combinations are checked. See
   [buffer acceptance](validation/buffers.md). No automatic deadlock recovery.

10. **Implemented and validated:** shared/per-machine quality-speed tables, exact
    scaling after base UPT, independent operation-keyed draws shared across modes,
    permanent defects with continued production, and final-only job inspection.
    Probabilities can be public/hidden. Fixed-label SPT/first-feasible require the
    label on all candidates. Hand makespans 24/20/16, preregistered probability
    checks and all existing module combinations replay through the same kernel.
    See [quality-speed acceptance](validation/quality-speed.md). No rework, quality
    CP, weighted objective, batch or learning is added.

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
- Logistics order is now agreed: item 8 fixed-matrix AGV with unlimited waiting
  areas, then item 9 finite buffers/reservations/blocking. Job location and vehicle
  ownership stay in the engine; capacity constraints must not create another writer.
- After item 10, study/batch composition, paired module ablations, single-host
  process concurrency/recovery and separate evidence/progress/debug policies are
  implemented and tested on small cases. See [study acceptance](validation/studies.md).
  The shared AGV holding buffer needed by the frozen IDETC source is implemented
  with finite/infinite capacity and exact replay; see [holding acceptance](validation/holding-buffer.md).
  Item 11 now has a frozen 4-case × 3-SPT-mode × 5-replication input bundle,
  conversion checks and a postcommit 60/60 replay/observation acceptance command.
  Actual integration completion is determined by its saved report, not development
  probes. See [IDETC integration](validation/idetc-integration.md). No learning,
  old-policy restoration, task-graph runtime edits or CP difficulty classification.
- Item 12 implements shared reveal-bound projection/Gym episodes, optional RLlib
  PPO and SB3 MaskablePPO training, and checkpoint policies through ordinary
  run/study. Development acceptance has actual fixed-budget updates/save/load and
  15/15 paired evaluations with exact replay; the final main-commit report remains
  the completion authority. See [learning acceptance](validation/learning.md).
  There is no performance-over-SPT requirement.
  Item 13's projection/coordination/PettingZoo checkpoint is committed as 467a31e.
  **Item 13 is implemented and formally validated on macOS:** clean integrated
  implementation `469c45f` passed fresh two-role training, save/load and 10/10
  paired evaluations with training/joint/action/schedule replay. The fixed
  seed101/4096-round recipe includes learner reward scale 0.0001; physics, masks,
  NOOP and raw episode rewards are unchanged. MARL mean 214.6 remains worse than
  SPT 134.6; no superiority is required. Earlier failed attempts remain retained.
  Training, evaluation and final audit identify the same clean implementation.
  Linux/h20 validation is deferred to Week3. See
  [resource MARL acceptance](validation/resource-marl.md).
  Profile environment, projection, evidence and training before further optimization.

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

Fixed-matrix transport with unlimited waiting areas precedes finite buffers,
destination reservations and blocking. Add route-level AGV
conflicts, charging, and traffic only after basic transport semantics are
validated. Introduce energy first as a metric/objective and add hard energy
constraints only when a research question requires them.

## M6 — Learning

Add a Gymnasium single-agent adapter before a DRL learner. Add W&B only as an
optional telemetry sink. PettingZoo and MARL use stable staged or joint
decision semantics and do not alter the canonical simulator core.
