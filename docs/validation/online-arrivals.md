# Week2 item 5 — Online arrivals and reveal-aware decisions

This slice adds independent arrival timing to the existing serial FJSP engine.
The contract is fixed in [ADR 0003](../decisions/0003-online-arrival-timing-and-visibility.md).
It adds no dependency, runtime sampling, dynamic CP, breakdown/repair, learning,
batch or solver runtime classification.

## Inputs and API

```python
from smartsom.domain import ArrivalPlan, JobArrival
from smartsom.dispatch import WaitNextEvent
from smartsom.engine import Simulator, replay, replay_schedule

arrivals = ArrivalPlan((JobArrival("A", 0, 0), JobArrival("B", 2, 1)))
sim = Simulator(factory, workload, arrivals=arrivals, decision_trigger="arrival_event")
# current_decision is pure; an arrival notification may have empty candidates.
# A caller can explicitly submit sim.step(WaitNextEvent()).
# replay(..., arrivals=arrivals, decision_trigger=...) uses the same step API.
```

Each plan explicitly covers all jobs, including those available at zero. Times
are nonnegative integers with reveal no later than release. Dispatch checks
physical release; snapshots expose full job descriptions only after reveal.
Factories and workload instances retain their existing schemas and digests.

Fixed scenario declaration (references resolve relative to the scenario):

```yaml
arrivals:
  kind: fixed
  path: ../../data/event_sets/online_arrivals.jsonl
decision_trigger: arrival_event
```

Each nonblank JSONL row is a strict `smartsom.job-arrival/v1` object:

```json
{"schema":"smartsom.job-arrival/v1","job_id":"B","release_at":2,"reveal_at":1}
```

Rows may include `provenance`, either null or the generator identity/version,
profile SHA-256 and effective seed. When present it must agree across all rows.
Repeating this small metadata object keeps every line a job record and the file
self-contained. Unknown fields, duplicate JSON keys, bad versions, invalid
times, duplicate IDs and incomplete/unknown job coverage fail during resolution.
Row order and whitespace do not affect the normalized arrival content digest;
the original byte digest is separately retained in manifest sources.

Generated scenario declaration:

```yaml
arrivals:
  kind: uniform_release_v1
  profile:
    initial_job_count: 0
    release_window: {min: 2, max: 5}
    notice_ticks: 1
decision_trigger: dispatch_available
```

The generator sorts job IDs lexically, initializes the first N at zero, and
draws one inclusive integer release per remaining job using a local RNG and the
derived `demand` seed. Reveal is `max(0, release - notice_ticks)`. N must be
between zero and the number of jobs; the release window has positive ordered
integer bounds; notice is a nonnegative integer. Fixed windows still consume a
draw. Off/fixed/all-initial runs consume no demand randomness. The workload RNG
is independent, and both old JSP/FJSP golden instances remain unchanged.

Golden case: run seed 42 derives demand seed **1575026194469689329**. For jobs
A/B and the profile above, the v1 timing rows are A=(release 2, reveal 1),
B=(release 3, reveal 2).

`dispatch_available` returns only legal dispatch choices. `arrival_event` also
returns once after reveal/release even without choices. SPT/first-feasible select
`WaitNextEvent()` on empty views. A scripted wait is explicitly serialized as
`{"kind":"wait_next_event"}`; `{}` is invalid. Existing dispatch and
`{"until": tick}` script shapes continue to work.

## Independent acceptance

The fixed hand case has A: M1/3 -> M2/2, release/reveal 0;
B: M2/2 -> M1/1, release 2 and reveal 1 (also tested at 0 and 2).

| Operation | Machine | Interval |
|---|---|---|
| A1 | M1 | [0, 3) |
| B1 | M2 | [2, 4) |
| A2 | M2 | [4, 6) |
| B2 | M1 | [4, 5) |

Makespan is **6**. At tick 1, `arrival_event` observes B's complete chain but
cannot dispatch it; SPT waits. The default trigger advances to tick 2. Both
produce these same intervals, although their decision/wait traces differ.
Replay of each run's own actions reproduces its complete trace. Canonical
schedule replay can order simultaneous starts differently from SPT; it checks
exact intervals, not equality of those two decision sequences.

[Frozen JobShopLib reference](../../data/reference/online_arrivals/sources.json)
uses source commit `460510f197744eed1cbcbbdfd6ec3252252f412c` and records a fresh
offline solve of the release-constrained case. The raw zero-indexed result,
source/result hashes, semantic schedule, and an optional reproduction script are
stored together. JobShopLib is not a project dependency. This reference checks
release feasibility; reveal isolation is established separately by engine tests.

Focused tests cover hand intervals, both triggers, zero-input equivalence,
no-lookahead changes to hidden worlds, immutable snapshots, atomic rejection,
wait interruption, no initial jobs, completion/reveal/release ordering, exact
idle-time replay, seeded generation, JSONL/schema errors, export/reimport,
working-directory/hash-seed independence and failed-run evidence. Earlier
static hand, ft06, Mk01 and official FJSP tests remain regression gates.

## Run evidence and verification

```bash
uv run smartsom validate configs/runs/online_arrivals_event.yaml
uv run smartsom run configs/runs/online_arrivals_dispatch.yaml
uv run smartsom run configs/runs/online_arrivals_event.yaml
uv run smartsom run configs/runs/generated_arrivals_dispatch.yaml
uv run smartsom run configs/runs/generated_arrivals_event.yaml
```

With arrivals enabled, `realized_events.jsonl` is reusable as a fixed table and
`observations.jsonl` contains every actual public snapshot, including the one
whose policy selection fails. The manifest records separate arrival content,
source-byte and profile digests, historical generation provenance, effective
seeds/consumption and decision trigger. No path, wall clock or run identity is
included in semantic trace. Static runs do not acquire the two extra files.

Quality gates: focused arrival tests, locked base environment/full pytest and
dependency check, locked CP environment/full pytest with `SMARTSOM_REQUIRE_CP=1`,
Ruff, formatting and `git diff --check`. The CP gate requires actual official
FJSP **6/6/6**, Mk01 **40/40/40** and ft06 **55/55/55** regressions; dynamic
scenarios are rejected by that provider, without fallback.

Pre-commit verification on 2026-09-07: **70 focused arrival tests passed**;
the actual base environment returned **354 passed, 7 skipped**, with PyJobShop,
OR-Tools and fjsplib absent; the restored locked CP environment returned
**361 passed** (two dependency-import deprecation warnings). Both environments
passed dependency consistency checks. Ruff, formatting and diff checks passed.

Post-commit acceptance lives under `artifacts/week2-item5/<commit-prefix>/`.
It records the integrated source identity and successful fixed/generated runs,
both decision triggers, scripted/action/schedule replay and export/reimport.
These generated run directories are not committed. The integration commit and
its fresh acceptance record, rather than older worktree results, establish
completion of this slice.
