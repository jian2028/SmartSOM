# Rule controller execution cost

Local engineering checks, 2026-10-24, starting from `6f02aaf` with the changes
in this working tree. These measurements are not a formal experiment or a
milestone acceptance. Simulation throughput and host execution time are separate.

## Changes

- Keep the public observations, full decision records and physical tick contract.
  Copy ordinary scalar values without recursive calls; shallow-copy a container
  only when every contained value is proven immutable. Preserve aliases, cycles,
  extension hooks and customized copy dispatch through the existing fallback.
- Use that same detached-copy contract for production snapshots. Serialize the
  concrete immutable `ProductionStep` fields directly, retaining the previous
  field names and tuple types. Subclasses and extension leaves retain `asdict`.
- In AGV corridor clearance, search for routes only after finding a nearby
  vehicle that could obstruct a port exit.
- Accept each AGV's preferred action directly when next cells are distinct,
  edges do not swap, no shared-port approaches can overlap during assignment,
  and no transit-port entry requires departure lookahead. Otherwise run the
  existing priority-inheritance/backtracking assignment. Safe following and
  longer rotations remain legal. Opposite station sides retain concurrency.

All routing rankings, dispatch priorities, service durations, inspection rules,
movement masks and collision arbitration retain their existing meaning. There
is no coarser time step, skipped decision evidence or fixed-layout assumption.
The optimization uses no new dependency and adds no checkpoint state.

## Profiling and paired timing

Template 7, 256 original jobs, world seed 202, policy seeds 11/12/13/14,
traffic admission, no active-fleet cap, headless, disk replay recording off.
The controller still constructs its full ordinary decision/replay records.

In the original 200-tick cProfile run, detached copies accounted for 181.34 of
288.10 coordinator seconds (63%), and public-view construction for 91.69 seconds
(32%). The latter includes snapshot construction. AGV planning occupied 2.37
seconds (less than 1%). Nested costs must not be added again. Profiler overhead
is substantial; use the separate unprofiled comparison below for timing.

| First 200 physical ticks of the same 256-job case | Before | After |
| --- | ---: | ---: |
| Coordinator wall seconds | 53.8325 | 26.2609 |
| Coordinator CPU seconds | 53.1250 | 25.8594 |
| Delivered jobs at tick 200 | 10 | 10 |

This pair measured 2.05 times the execution speed (51.2% less coordinator time).
It is one local timing pair, not a cross-platform or full-run speedup claim.
The before run restored the exact pre-edit copier and planner and the original
snapshot and step serialization behavior in a separate process. Canonical hashes
of the entire committed 200-tick traces match:
`2ec4f63301c03bbfc836414c61bed1d03e19861f5b79d2190b68e17dcda9e15e`.
This comparison includes decisions, observations, actions, states, events,
rejections and rewards. Timing excludes the external trace hashing.

The shortcut handled 31 of these boundaries; the full assignment handled 169.
Most of the measured improvement therefore comes from copying and serialization,
not bypassing traffic handling.

## Full 256-job rerun

The fresh optimized run completed 256/256 original jobs at tick 4,276, giving
59.869 jobs per 1,000 physical ticks. Every individual shipment time and every
per-tick delivery count exactly matches the prior 256-job reference. Moves
(28,924), pickup/drop port counts, AGV service/utilization counters and final
metrics also match. All 256 shipments pass the privileged final-quality check.
Persistent reversals, detected longer cycles and rejected proposals are zero.

Simulation execution inside the coordinator took 498.28 wall seconds
(8 minutes 18 seconds), with 491.17 process CPU seconds. The full verification
harness took 603.175 seconds including canonical hashing of every committed
record and its diagnostic loop. The extra comparison hashing is outside normal
controller execution. There is no fresh full-run baseline timing, so the 2.05
speed ratio above applies specifically to the paired 200-tick measurement.

Of 4,276 boundaries, 656 used the independent-action shortcut and 3,620 used the
existing coordinated assignment. These counters belong to the measurement
harness, not checkpoint or simulator state. The optimized runtime source hashes
match between the paired short run and the full run. The saved before-source
copies match the starting Git commit after newline normalization.

Raw profiles, before-source copies, measurement scripts and comparison JSON are
local under `E:\USC\IMPACT\.smartsom-controller-speed-20261024`; generated evidence
is not committed. `verified-comparison.json` records the exact equality checks
and the timing scopes. A readable copy of this note is saved as
`E:\USC\IMPACT\SmartSOM Controller Speed Report.md`.

## Verification scope

- 423 focused tests passed across boundary copying, AGV planning/admission,
  composition, persistent intentions, liveness, inspection routing, matrix
  transport, runtime recording/replay, machine reliability, structural snapshots,
  rule registration and four-file/three-map integration. Initial temporary-path
  and hard-link permission errors were resolved using a writable test directory
  and a scoped run of the three file-save checks.
- The planner shortcut matches full assignment on 60 varied synthetic public
  boundaries; targeted regressions cover merges, swaps, rotations, transit
  clearance and both station sides. These are engineering checks, not evidence
  of universal deadlock freedom.
- Fresh 12-job engineering probes finished on Templates 7/8/9 at ticks
  269/201/183, respectively, with no rejected AGV proposals, persistent reversals
  or detected longer cycles.
- Repository Ruff lint and format checks passed. CI ownership now explicitly
  includes the four existing controller test files; full-plan inventory
  validation passed.

The required Linux/macOS native CI union has not run on this Windows host.
Generating a full CI plan is not full CI execution. No learning speedup, disk
replay-export speedup, fault-response performance or general steady-state
throughput improvement is claimed by these checks.
