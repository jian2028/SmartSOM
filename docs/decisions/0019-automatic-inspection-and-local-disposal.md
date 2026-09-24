# 0019 — Automatic per-job inspection and local disposal

Date: 2026-09-24

Supersedes ADR 0017's controlled, station-locking inspection behavior. Other grid
transition, demand identity and recording contracts remain unchanged.

## Decision

Inspection is an automatic service, not a policy role. At a committed arrival
boundary t, an UNKNOWN job begins inspection if parallel capacity is available.
It completes after inspection_ticks elapsed transitions (t+2 for the default).
Each job has an independent timer. Busy jobs cannot be picked up; they do not
lock other jobs or prevent admission to free capacity. PASS remains resident.
Unlinked FAIL remains resident and can be collected by an AGV. Pending UNKNOWN
jobs beyond parallel capacity wait FIFO; max means the sum of slot capacities.
Completed jobs still consume storage capacity until removed.

An inspection station may specify auto_disposal_bin_id. Its target must be an
existing unlimited-capacity scrap bin sharing a cell edge with its footprint.
A linked FAIL enters DISPOSING for one further tick, still occupying its slot
and unavailable to AGVs. It then leaves the station, increments scrap metrics
and creates exactly one replacement attempt through the existing demand path.
The disposal requires no port, AGV or policy action. Shared targets accept
simultaneous disposals deterministically. Finite bins remain supported for
manual AGV disposal but cannot be linked for automatic disposal.

Single-cell inspection slots may hold multiple jobs. This is physical capacity,
not multiple resources sharing a grid cell. Studio displays four places as a
2×2 subdivision and preserves individual illustration timers separately from
simulation state.

## Compatibility and evidence

Factory v2 adds an optional reference defaulting to None; old factories load
without geometry or capacity changes, but execute with automatic inspection.
JointCommand retains its legacy quality field only to issue an explicit error
for nonempty START/WAIT commands. Learning action and observation contracts are
v2; v1 checkpoints require retraining. No quality actor/critic is created.
Each automatic inspection start retains the 0.1 inspection cost per job;
quality-reveal and replacement accounting are unchanged.

New snapshots include per-job inspection/disposal phases and remaining ticks.
Summary batch fields remain for inspection statistics and legacy display code;
they do not control execution. Old recorded batch frames remain viewable without
reexecution. Old command traces and learned weights are not silently migrated.
New disposal events drive replay transitions; Studio annotations never advance
simulation or generate replacement orders.

Hand-calculated transition tests, reference validation, editor round trips,
learning adapters and replay rendering are engineering checks, not experimental
performance acceptance.
