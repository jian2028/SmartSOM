# 0067 — Dispatcher trip transit time ratio

Status: accepted definition; implementation verification is separate.

Extends ADR 0060's passive research metrics. This authorizes local engineering
verification, not formal experiments, historical reruns or changes to physics,
observations, policies, rewards or random sequences.

## Trip and clock

A trip begins when a dispatcher destination reservation is accepted, at the
vehicle's current physical cell and tick. Rejected requests do not start trips.
It ends on the first arrival at any operation-compatible port of that owner.
Empty pickup and loaded delivery are separate groups; their combined result
weights completed trips equally. Empty trips need not have an assigned job yet.

The numerator is physical end tick minus acceptance tick. It includes waiting,
parking and detours before arrival, and excludes service/processing waits after
arrival. Traffic replanning for the same owner does not restart the clock.
Cancellation and destination replacement end the old trip with separate statuses;
a new acceptance starts another trip. Unassigned repositioning is outside scope.

The denominator is the static shortest feasible transit time from the accepted
start cell to the owner's frozen legal port set. Obstacles and equipment
footprints are included, transient vehicle occupancy is not. The current kernel
allows four-way movement and requires all four headings at active ports, one edge
per physical tick and no acceleration accumulator. Therefore shortest graph edge
count equals ideal transit ticks. A committed move event at tick t arrives at
t+1: a two-edge trip accepted at 7 and moving at 7 and 8 arrives at 9, yielding
(9-7)/2 = 1. Future speed or directed-route physics must revise this definition
using that kernel's feasible transit-time model; Manhattan distance is not used.

The initial selected port is not an immutable target: the existing traffic layer
can select another port of the same owner. The denominator consequently refers to
the legal goal set, not the temporary initial port or a retrospectively selected
endpoint. This also captures the cost of choosing a farther legal port.

## Missingness and aggregation

Only completed trips with finite positive ideal time contribute to the primary
arithmetic mean of per-trip elapsed/ideal ratios. Zero-distance arrivals complete
at acceptance with a missing ratio; unreachable baselines remain missing. Neither
is changed to 1, 0 or infinity. Separate counts retain completion, cancellation,
retargeting, censoring, zero distance and unreachable baselines.

An unfinished trip is censored at the actual episode endpoint. Its accrued elapsed
time and elapsed/ideal lower bound are recorded separately, never included in the
completed ratio mean. Censored sums include a count of valid positive baselines.
The auxiliary completed sum(elapsed)/sum(ideal) is an ideal-time-weighted ratio,
not the arithmetic trip mean. The nested counters retain sufficient statistics.

Each case/episode has all/empty/loaded summaries. Existing evaluation hierarchy
averages repeats within a world, worlds within a condition, then conditions within
a training seed. Research reports then average training seeds equally, with
actual/planned seed coverage and sample SD. Trips or worlds from multiple training
seeds are not pooled as independent seed observations. A case with no valid
completed trip has a missing mean; missing means are not silently zero-filled.
Aggregated count fields are equal-weight case/world means, not pooled totals;
exact integer counts remain in each case. Historical rows stay missing.

## Storage and dependency boundaries

`TripTransitMetrics` consumes ordered committed events and static routes as a
passive telemetry observer. No engine-to-telemetry import is added. The observer
tracks O(vehicles) active trips and counters. Static goal-set distance maps reuse
the route cache (one O(V+E) BFS per new goal set, then O(1) start lookups); there is
no per-tick BFS. Each move and lifecycle event needs O(1) tracking apart from the
first cache miss. Default summary/detailed metrics persist aggregates only.

Only full recording writes `replay/trips.jsonl`, referenced by `run.json` and bound
to that episode's scenario/source and containing per-AGV sequence, kind, accepted
job if known, owner/operation/goals, start/end cells and ticks, ideal/elapsed ticks,
status/reason, completed ratio or censored lower bound. Trip rows are streamed and
are not held in an episode-sized list or copied into cloud tables. Explicit
metrics=off disables this observer. Centralized assignments do not emit dispatcher
acceptance events and are marked not applicable rather than reported as zero.

Verification includes independent hand paths for ideal=1, waiting, obstacle-aware
shortest routes and detours, zero/unreachable, cancellation/replacement/replan,
censoring, arithmetic versus weighted means, unequal world/seed panels, real
kernel tick ordering, full recording and unchanged actions/events/rewards/RNG.
Measured runtime overhead and the complete integrated CI remain separate gates.
